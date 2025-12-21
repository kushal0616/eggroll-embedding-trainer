#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm
import logging

from src.data.load_nanobeir import NanoBEIRLoader
from src.data.build_pools import PoolBuilder, PoolConfig
from src.data.cache_prehead import EmbeddingCache
from src.eggroll.noise import Rank1NoiseGenerator, NoiseConfig
from src.eggroll.scoring import VectorizedScorer
from src.eggroll.ndcg import CachedNDCGComputer
from src.eggroll.shaping import FitnessShaper, AntitheticShaper
from src.eggroll.update import EGGROLLUpdater, UpdateConfig
from src.baselines.contrastive_trainer import ContrastiveTrainer, BaselineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_ndcg(W, H_q, H_d, rel, k=10, device="mps"):
    q = H_q @ W.T
    d = torch.einsum('bpd,od->bpo', H_d, W)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    scores = torch.einsum('bd,bpd->bp', q, d)
    
    _, topk_idx = scores.topk(k, dim=1)
    topk_rel = torch.gather(rel, dim=1, index=topk_idx)
    
    positions = torch.arange(k, device=device)
    discounts = 1.0 / torch.log2(positions + 2)
    gains = (2.0 ** topk_rel.float()) - 1.0
    dcg = (gains * discounts).sum(dim=1)
    
    sorted_rel, _ = rel.sort(dim=-1, descending=True)
    ideal_gains = (2.0 ** sorted_rel[:, :k].float()) - 1.0
    idcg = (ideal_gains * discounts).sum(dim=1).clamp(min=1e-10)
    
    return (dcg / idcg).mean().item()


def run_eggroll(
    H_q_train, H_d_train, rel_train,
    H_q_val, H_d_val, rel_val,
    num_steps=1000,
    device="mps",
    sigma=0.05,
    population_size=256,
    learning_rate=0.1,
    D_out=128
):
    D_in = H_q_train.shape[1]
    
    W_cpu = torch.empty(D_out, D_in)
    torch.nn.init.orthogonal_(W_cpu)
    W = W_cpu.to(device)
    
    noise_config = NoiseConfig(
        rank=1, 
        population_size=population_size, 
        sigma=sigma, 
        seed=42
    )
    noise_gen = Rank1NoiseGenerator(
        noise_config, 
        output_size=D_out, 
        input_size=D_in, 
        device=device
    )
    scorer = VectorizedScorer(sigma=sigma)
    ndcg_computer = CachedNDCGComputer(k=10, device=device)
    shaper = FitnessShaper(method='rank')
    anti = AntitheticShaper()
    update_config = UpdateConfig(learning_rate=learning_rate, clip_norm=0.5)
    updater = EGGROLLUpdater(
        update_config, 
        output_size=D_out, 
        input_size=D_in, 
        device=device
    )
    
    history = {"train": [], "val": [], "step": []}
    best_val = 0
    
    pbar = tqdm(range(num_steps), desc="EGGROLL")
    for step in pbar:
        A, B = noise_gen.sample(step=step)
        scores = scorer.compute_all_scores(H_q_train, H_d_train, W, A, B)
        ndcg_mean, ndcg_per_query = ndcg_computer.compute_ndcg(scores, rel_train)
        shaped = shaper(ndcg_mean, ndcg_per_query)
        delta = anti(shaped)
        W = updater.apply_update(W, A, B, delta, sigma)
        
        if step % 100 == 0:
            train_ndcg = compute_ndcg(W, H_q_train, H_d_train, rel_train, device=device)
            val_ndcg = compute_ndcg(W, H_q_val, H_d_val, rel_val, device=device)
            best_val = max(best_val, val_ndcg)
            
            history["train"].append(train_ndcg)
            history["val"].append(val_ndcg)
            history["step"].append(step)
            
            pbar.set_postfix({"train": f"{train_ndcg:.4f}", "val": f"{val_ndcg:.4f}"})
    
    final_train = compute_ndcg(W, H_q_train, H_d_train, rel_train, device=device)
    final_val = compute_ndcg(W, H_q_val, H_d_val, rel_val, device=device)
    
    return {
        "final_train": final_train,
        "final_val": final_val,
        "best_val": best_val,
        "history": history,
        "W": W
    }


def run_baseline_contrastive(
    task="NanoMSMARCO",
    num_steps=1000,
    learning_rate=1e-3,
    device="mps"
):
    config = BaselineConfig(
        task=task,
        num_steps=num_steps,
        learning_rate=learning_rate,
        device=device,
        pool_size=64,
        head_output_size=128
    )
    trainer = ContrastiveTrainer(config)
    return trainer.train()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="NanoMSMARCO")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    
    device = args.device
    task = args.task
    num_steps = args.num_steps
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    logger.info(f"Loading data for {task}...")
    
    cache = EmbeddingCache(Path("./cache"), device=device)
    query_cache = cache.load(f"{task}_queries")
    doc_cache = cache.load(f"{task}_corpus")
    
    loader = NanoBEIRLoader(task)
    data = loader.load_all()
    train_ids, val_ids = loader.split_queries(list(data['queries'].keys()), 0.8)
    
    pool_config = PoolConfig(pool_size=64, pos_cap=3, rand_count=8)
    builder = PoolBuilder(pool_config)
    train_pools = builder.build_all_pools(
        train_ids, data['qrels'], data['bm25'], 
        list(data['corpus'].keys()), seed=42
    )
    val_pools = builder.build_all_pools(
        val_ids, data['qrels'], data['bm25'], 
        list(data['corpus'].keys()), seed=43
    )
    
    H_q_train = query_cache.get_embeddings(train_ids)
    H_d_train = torch.stack([
        doc_cache.get_embeddings(train_pools[qid].doc_ids) 
        for qid in train_ids
    ])
    rel_train = torch.stack([
        torch.tensor(train_pools[qid].relevance, device=device) 
        for qid in train_ids
    ])
    
    H_q_val = query_cache.get_embeddings(val_ids)
    H_d_val = torch.stack([
        doc_cache.get_embeddings(val_pools[qid].doc_ids) 
        for qid in val_ids
    ])
    rel_val = torch.stack([
        torch.tensor(val_pools[qid].relevance, device=device) 
        for qid in val_ids
    ])
    
    logger.info(f"Train: {len(train_ids)} queries, Val: {len(val_ids)} queries")
    logger.info(f"Pool size: {H_d_train.shape[1]}")
    
    print("\n" + "="*60)
    print("EXPERIMENT: EGGROLL vs Baseline Contrastive")
    print("="*60)
    
    print("\n[1/2] Running EGGROLL (ES-based, directly optimizes ranking)...")
    eggroll_results = run_eggroll(
        H_q_train, H_d_train, rel_train,
        H_q_val, H_d_val, rel_val,
        num_steps=num_steps,
        device=device,
        sigma=0.05,
        population_size=256,
        learning_rate=0.1
    )
    
    print("\n[2/2] Running Baseline (Gradient-based contrastive loss)...")
    baseline_results = run_baseline_contrastive(
        task=task,
        num_steps=num_steps,
        learning_rate=1e-3,
        device=device
    )
    
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    print(f"\n{'Method':<25} {'Train NDCG@10':<15} {'Val NDCG@10':<15} {'Best Val':<15}")
    print("-"*70)
    print(f"{'EGGROLL (ES)':<25} {eggroll_results['final_train']:<15.4f} {eggroll_results['final_val']:<15.4f} {eggroll_results['best_val']:<15.4f}")
    print(f"{'Baseline (Contrastive)':<25} {baseline_results['final_train_ndcg']:<15.4f} {baseline_results['final_val_ndcg']:<15.4f} {baseline_results['best_val_ndcg']:<15.4f}")
    
    eggroll_improvement = (eggroll_results['final_train'] / 0.22 - 1) * 100
    baseline_improvement = (baseline_results['final_train_ndcg'] / 0.22 - 1) * 100
    
    print(f"\nImprovement from random init (~0.22):")
    print(f"  EGGROLL:  +{eggroll_improvement:.1f}%")
    print(f"  Baseline: +{baseline_improvement:.1f}%")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
