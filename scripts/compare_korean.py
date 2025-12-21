#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm
import logging

from src.data.load_ko_strategyqa import KoStrategyQALoader
from src.data.build_pools import PoolBuilder, PoolConfig
from src.data.cache_prehead import EmbeddingCache
from src.model.multilingual_encoder import MultilingualE5Encoder
from src.eggroll.noise import Rank1NoiseGenerator, NoiseConfig
from src.eggroll.scoring import VectorizedScorer
from src.eggroll.ndcg import CachedNDCGComputer
from src.eggroll.shaping import FitnessShaper, AntitheticShaper
from src.eggroll.update import EGGROLLUpdater, UpdateConfig

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


def run_eggroll(H_q_train, H_d_train, rel_train, H_q_val, H_d_val, rel_val,
                num_steps=1000, device="mps", sigma=0.05, pop_size=256, lr=0.1, D_out=128):
    D_in = H_q_train.shape[1]
    
    W_cpu = torch.empty(D_out, D_in)
    torch.nn.init.orthogonal_(W_cpu)
    W = W_cpu.to(device)
    
    noise_config = NoiseConfig(rank=1, population_size=pop_size, sigma=sigma, seed=42)
    noise_gen = Rank1NoiseGenerator(noise_config, output_size=D_out, input_size=D_in, device=device)
    scorer = VectorizedScorer(sigma=sigma)
    ndcg_computer = CachedNDCGComputer(k=10, device=device)
    shaper = FitnessShaper(method='rank')
    anti = AntitheticShaper()
    update_config = UpdateConfig(learning_rate=lr, clip_norm=0.5)
    updater = EGGROLLUpdater(update_config, output_size=D_out, input_size=D_in, device=device)
    
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
            pbar.set_postfix({"train": f"{train_ndcg:.4f}", "val": f"{val_ndcg:.4f}"})
    
    final_train = compute_ndcg(W, H_q_train, H_d_train, rel_train, device=device)
    final_val = compute_ndcg(W, H_q_val, H_d_val, rel_val, device=device)
    
    return {"final_train": final_train, "final_val": final_val, "best_val": max(best_val, final_val)}


def run_baseline(H_q_train, H_d_train, rel_train, H_q_val, H_d_val, rel_val,
                 num_steps=1000, device="mps", lr=1e-3, D_out=128):
    import torch.nn.functional as F
    
    D_in = H_q_train.shape[1]
    W = torch.nn.Parameter(torch.empty(D_out, D_in, device=device))
    torch.nn.init.orthogonal_(W.data.cpu())
    W.data = W.data.to(device)
    
    optimizer = torch.optim.AdamW([W], lr=lr, weight_decay=1e-4)
    temperature = 0.05
    
    best_val = 0
    pbar = tqdm(range(num_steps), desc="Baseline")
    
    for step in pbar:
        q = H_q_train @ W.T
        d = torch.einsum('bpd,od->bpo', H_d_train, W)
        q = F.normalize(q, dim=-1)
        d = F.normalize(d, dim=-1)
        
        scores = torch.einsum('bd,bpd->bp', q, d) / temperature
        pos_mask = (rel_train > 0).float()
        
        if pos_mask.sum() > 0:
            log_softmax = F.log_softmax(scores, dim=-1)
            loss = -(log_softmax * pos_mask).sum(dim=-1) / pos_mask.sum(dim=-1).clamp(min=1)
            loss = loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W], 1.0)
            optimizer.step()
        
        if step % 100 == 0:
            with torch.no_grad():
                train_ndcg = compute_ndcg(W.data, H_q_train, H_d_train, rel_train, device=device)
                val_ndcg = compute_ndcg(W.data, H_q_val, H_d_val, rel_val, device=device)
                best_val = max(best_val, val_ndcg)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "train": f"{train_ndcg:.4f}", "val": f"{val_ndcg:.4f}"})
    
    with torch.no_grad():
        final_train = compute_ndcg(W.data, H_q_train, H_d_train, rel_train, device=device)
        final_val = compute_ndcg(W.data, H_q_val, H_d_val, rel_val, device=device)
    
    return {"final_train": final_train, "final_val": final_val, "best_val": max(best_val, final_val)}


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max_queries", type=int, default=100)
    args = parser.parse_args()
    
    device = args.device
    torch.manual_seed(42)
    np.random.seed(42)
    
    cache_dir = Path("./cache")
    cache_dir.mkdir(exist_ok=True)
    
    query_cache_file = cache_dir / "KoStrategyQA_queries_embeddings.pt"
    corpus_cache_file = cache_dir / "KoStrategyQA_corpus_embeddings.pt"
    
    logger.info("Loading Ko-StrategyQA dataset...")
    loader = KoStrategyQALoader(max_corpus=3000)
    data = loader.load_all()
    
    query_ids = list(data['queries'].keys())[:args.max_queries]
    data['queries'] = {qid: data['queries'][qid] for qid in query_ids}
    data['qrels'] = {qid: data['qrels'][qid] for qid in query_ids if qid in data['qrels']}
    
    logger.info(f"Queries: {len(data['queries'])}, Corpus: {len(data['corpus'])}")
    
    if not query_cache_file.exists() or not corpus_cache_file.exists():
        logger.info("Loading multilingual-e5-base encoder...")
        encoder = MultilingualE5Encoder(pooling='mean')
        encoder.to(device)
        encoder.freeze()
        encoder.eval()
        
        logger.info("Caching query embeddings...")
        query_texts = list(data['queries'].values())
        query_ids_list = list(data['queries'].keys())
        
        query_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(query_texts), 16), desc="Encoding queries"):
                batch = query_texts[i:i+16]
                emb = encoder.encode(batch, is_query=True)
                query_embeddings.append(emb.cpu())
        query_embeddings = torch.cat(query_embeddings, dim=0).half()
        
        torch.save({
            "ids": query_ids_list,
            "embeddings": query_embeddings,
            "id_to_idx": {id_: idx for idx, id_ in enumerate(query_ids_list)}
        }, query_cache_file)
        
        logger.info("Caching corpus embeddings...")
        corpus_texts = list(data['corpus'].values())
        corpus_ids_list = list(data['corpus'].keys())
        
        corpus_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(corpus_texts), 16), desc="Encoding corpus"):
                batch = corpus_texts[i:i+16]
                emb = encoder.encode(batch, is_query=False)
                corpus_embeddings.append(emb.cpu())
        corpus_embeddings = torch.cat(corpus_embeddings, dim=0).half()
        
        torch.save({
            "ids": corpus_ids_list,
            "embeddings": corpus_embeddings,
            "id_to_idx": {id_: idx for idx, id_ in enumerate(corpus_ids_list)}
        }, corpus_cache_file)
        
        del encoder
        torch.mps.empty_cache() if device == "mps" else None
    
    logger.info("Loading cached embeddings...")
    cache = EmbeddingCache(cache_dir, device=device)
    query_cache = cache.load("KoStrategyQA_queries")
    doc_cache = cache.load("KoStrategyQA_corpus")
    
    train_ids, val_ids = loader.split_queries(list(data['queries'].keys()), 0.8)
    logger.info(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
    
    pool_config = PoolConfig(pool_size=64, pos_cap=3, rand_count=16)
    builder = PoolBuilder(pool_config)
    
    corpus_ids = list(data['corpus'].keys())
    train_pools = builder.build_all_pools(train_ids, data['qrels'], data['bm25'], corpus_ids, seed=42)
    val_pools = builder.build_all_pools(val_ids, data['qrels'], data['bm25'], corpus_ids, seed=43)
    
    H_q_train = query_cache.get_embeddings(train_ids)
    H_d_train = torch.stack([doc_cache.get_embeddings(train_pools[qid].doc_ids) for qid in train_ids])
    rel_train = torch.stack([torch.tensor(train_pools[qid].relevance, device=device) for qid in train_ids])
    
    H_q_val = query_cache.get_embeddings(val_ids)
    H_d_val = torch.stack([doc_cache.get_embeddings(val_pools[qid].doc_ids) for qid in val_ids])
    rel_val = torch.stack([torch.tensor(val_pools[qid].relevance, device=device) for qid in val_ids])
    
    print("\n" + "="*70)
    print("KOREAN IR EXPERIMENT: EGGROLL vs Baseline (Ko-StrategyQA)")
    print("="*70)
    print(f"Dataset: Ko-StrategyQA, Train: {len(train_ids)}, Val: {len(val_ids)}")
    print(f"Encoder: multilingual-e5-base (768-dim)")
    
    init_train = compute_ndcg(
        torch.randn(128, 768, device=device) * 0.01,
        H_q_train, H_d_train, rel_train, device=device
    )
    print(f"Random init baseline NDCG: ~{init_train:.4f}")
    
    print("\n[1/2] Running EGGROLL...")
    eggroll_results = run_eggroll(
        H_q_train, H_d_train, rel_train,
        H_q_val, H_d_val, rel_val,
        num_steps=args.num_steps,
        device=device
    )
    
    print("\n[2/2] Running Baseline (Contrastive)...")
    baseline_results = run_baseline(
        H_q_train, H_d_train, rel_train,
        H_q_val, H_d_val, rel_val,
        num_steps=args.num_steps,
        device=device
    )
    
    print("\n" + "="*70)
    print("RESULTS: Ko-StrategyQA (Korean IR)")
    print("="*70)
    print(f"\n{'Method':<25} {'Train NDCG@10':<15} {'Val NDCG@10':<15} {'Best Val':<15}")
    print("-"*70)
    print(f"{'EGGROLL (ES)':<25} {eggroll_results['final_train']:<15.4f} {eggroll_results['final_val']:<15.4f} {eggroll_results['best_val']:<15.4f}")
    print(f"{'Baseline (Contrastive)':<25} {baseline_results['final_train']:<15.4f} {baseline_results['final_val']:<15.4f} {baseline_results['best_val']:<15.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
