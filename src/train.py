import torch
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import logging
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.data.load_nanobeir import NanoBEIRLoader
from src.data.build_pools import PoolBuilder, PoolConfig, QueryPool
from src.data.cache_prehead import EmbeddingCache, LoadedCache
from src.model.encoder import ModernBERTEncoder
from src.model.head import ProjectionHead
from src.eggroll.noise import Rank1NoiseGenerator, NoiseConfig
from src.eggroll.scoring import VectorizedScorer
from src.eggroll.ndcg import CachedNDCGComputer
from src.eggroll.shaping import FitnessShaper, AntitheticShaper
from src.eggroll.update import EGGROLLUpdater, UpdateConfig, AdaptiveSigmaUpdater
from src.utils.device import get_device

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    task: str = "NanoMSMARCO"
    train_ratio: float = 0.8
    
    pool_size: int = 256
    pos_cap: int = 3
    rand_count: int = 16
    pool_refresh_interval: int = 500
    
    hidden_size: int = 768
    head_output_size: int = 256
    pooling: str = "mean"
    
    population_size: int = 256
    rank: int = 1
    sigma: float = 0.02
    adaptive_sigma: bool = True
    
    ndcg_k: int = 20
    
    learning_rate: float = 0.05
    clip_norm: float = 1.0
    weight_decay: float = 1e-4
    momentum: float = 0.0
    
    num_steps: int = 5000
    eval_interval: int = 100
    full_eval_interval: int = 500
    log_interval: int = 10
    
    shaping_method: str = "rank"
    
    device: str = "auto"
    seed: int = 42
    cache_dir: str = "./cache"
    checkpoint_dir: str = "./checkpoints"
    
    use_wandb: bool = True
    project_name: str = "eggroll-ndcg"


class EGGROLLTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = get_device(config.device)
        self.step = 0
        
        logger.info(f"Using device: {self.device}")
        torch.manual_seed(config.seed)
        
        self._init_data()
        self._init_model()
        self._init_eggroll()
        self._init_logging()
        
    def _init_data(self):
        cfg = self.config
        logger.info(f"Loading NanoBEIR task: {cfg.task}")
        
        loader = NanoBEIRLoader(cfg.task, seed=cfg.seed)
        data = loader.load_all()
        
        self.queries = data["queries"]
        self.corpus = data["corpus"]
        self.qrels = data["qrels"]
        self.bm25 = data["bm25"]
        
        all_query_ids = list(self.queries.keys())
        self.train_query_ids, self.val_query_ids = loader.split_queries(
            all_query_ids, cfg.train_ratio
        )
        logger.info(f"Train queries: {len(self.train_query_ids)}, Val: {len(self.val_query_ids)}")
        
        pool_config = PoolConfig(
            pool_size=cfg.pool_size,
            pos_cap=cfg.pos_cap,
            rand_count=cfg.rand_count
        )
        pool_builder = PoolBuilder(pool_config)
        
        corpus_ids = list(self.corpus.keys())
        self.train_pools = pool_builder.build_all_pools(
            self.train_query_ids, self.qrels, self.bm25, corpus_ids, seed=cfg.seed
        )
        self.val_pools = pool_builder.build_all_pools(
            self.val_query_ids, self.qrels, self.bm25, corpus_ids, seed=cfg.seed + 1
        )
        
        self._init_cache()
        
    def _init_cache(self):
        cfg = self.config
        cache = EmbeddingCache(Path(cfg.cache_dir), device=self.device)
        query_cache_file = Path(cfg.cache_dir) / f"{cfg.task}_queries_embeddings.pt"
        corpus_cache_file = Path(cfg.cache_dir) / f"{cfg.task}_corpus_embeddings.pt"
        
        if query_cache_file.exists() and corpus_cache_file.exists():
            logger.info("Loading cached embeddings...")
            self.query_cache = cache.load(f"{cfg.task}_queries")
            self.doc_cache = cache.load(f"{cfg.task}_corpus")
        else:
            logger.info("Computing embeddings (this may take a while)...")
            encoder = ModernBERTEncoder(pooling=cfg.pooling)
            encoder.to(self.device)
            encoder.freeze()
            encoder.eval()
            
            cache.compute_and_save(encoder, self.queries, task_name=f"{cfg.task}_queries")
            cache.compute_and_save(encoder, self.corpus, task_name=f"{cfg.task}_corpus")
            
            self.query_cache = cache.load(f"{cfg.task}_queries")
            self.doc_cache = cache.load(f"{cfg.task}_corpus")
            
            del encoder
            torch.cuda.empty_cache()
            
    def _init_model(self):
        cfg = self.config
        self.head = ProjectionHead(
            input_size=cfg.hidden_size, 
            output_size=cfg.head_output_size,
            normalize=False
        )
        self.W = self.head.W.data.to(self.device)
        
    def _init_eggroll(self):
        cfg = self.config
        
        noise_config = NoiseConfig(
            rank=cfg.rank,
            population_size=cfg.population_size,
            sigma=cfg.sigma,
            seed=cfg.seed
        )
        self.noise_gen = Rank1NoiseGenerator(
            noise_config, 
            output_size=cfg.head_output_size,
            input_size=cfg.hidden_size,
            device=self.device
        )
        
        self.scorer = VectorizedScorer(sigma=cfg.sigma)
        self.ndcg_computer = CachedNDCGComputer(k=cfg.ndcg_k, device=self.device)
        self.fitness_shaper = FitnessShaper(method=cfg.shaping_method)
        self.antithetic_shaper = AntitheticShaper()
        
        update_config = UpdateConfig(
            learning_rate=cfg.learning_rate,
            clip_norm=cfg.clip_norm,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum
        )
        self.updater = EGGROLLUpdater(
            update_config, 
            output_size=cfg.head_output_size,
            input_size=cfg.hidden_size,
            device=self.device
        )
        
        if cfg.adaptive_sigma:
            self.sigma_adapter = AdaptiveSigmaUpdater(initial_sigma=cfg.sigma)
        else:
            self.sigma_adapter = None
            
    def _init_logging(self):
        cfg = self.config
        if cfg.use_wandb and WANDB_AVAILABLE:
            wandb.init(project=cfg.project_name, config=vars(cfg))
            
    def get_batch(self, query_ids: List[str], pools: Dict[str, QueryPool]):
        batch_query_ids = query_ids
        
        H_q = self.query_cache.get_embeddings(batch_query_ids)
        
        doc_ids_batch = [pools[qid].doc_ids for qid in batch_query_ids]
        H_d = torch.stack([
            self.doc_cache.get_embeddings(doc_ids)
            for doc_ids in doc_ids_batch
        ])
        
        relevance = torch.stack([
            torch.tensor(pools[qid].relevance, device=self.device)
            for qid in batch_query_ids
        ])
        
        return H_q, H_d, relevance, batch_query_ids
    
    def train_step(self):
        cfg = self.config
        
        H_q, H_d, relevance, query_ids = self.get_batch(
            self.train_query_ids, self.train_pools
        )
        
        A, B = self.noise_gen.sample(step=self.step)
        
        sigma = self.sigma_adapter.sigma if self.sigma_adapter else cfg.sigma
        self.scorer.sigma = sigma
        
        scores = self.scorer.compute_all_scores(H_q, H_d, self.W, A, B)
        
        ndcg_mean, ndcg_per_query = self.ndcg_computer.compute_ndcg(scores, relevance)
        
        shaped_mean = self.fitness_shaper(ndcg_mean, ndcg_per_query)
        delta_fitness = self.antithetic_shaper(shaped_mean)
        
        W_old = self.W.clone()
        self.W = self.updater.apply_update(self.W, A, B, delta_fitness, sigma)
        delta_W = self.W - W_old
        
        if self.sigma_adapter:
            M = cfg.population_size // 2
            self.sigma_adapter.update(ndcg_mean[:M], ndcg_mean[M:])
        
        metrics = {
            "train/ndcg_mean": ndcg_mean.mean().item(),
            "train/ndcg_std": ndcg_mean.std().item(),
            "train/sigma": sigma,
            "train/update_norm": torch.norm(delta_W, p='fro').item(),
            "train/weight_norm": torch.norm(self.W, p='fro').item(),
        }
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, split: str = "val"):
        cfg = self.config
        
        if split == "val":
            query_ids = self.val_query_ids
            pools = self.val_pools
        else:
            query_ids = self.train_query_ids
            pools = self.train_pools
            
        if not query_ids:
            return {f"{split}/ndcg@{cfg.ndcg_k}": 0.0}
            
        all_ndcg = []
        
        batch_size = 64
        for i in range(0, len(query_ids), batch_size):
            batch_ids = query_ids[i:i + batch_size]
            
            H_q = self.query_cache.get_embeddings(batch_ids)
            H_d = torch.stack([
                self.doc_cache.get_embeddings(pools[qid].doc_ids)
                for qid in batch_ids
            ])
            relevance = torch.stack([
                torch.tensor(pools[qid].relevance, device=self.device)
                for qid in batch_ids
            ])
            
            q = H_q @ self.W.T
            d = torch.einsum('bpd,ed->bpe', H_d, self.W)
            scores = torch.einsum('bd,bpd->bp', q, d)
            
            k = min(cfg.ndcg_k, scores.shape[1])
            _, topk_idx = scores.topk(k, dim=1)
            topk_rel = torch.gather(relevance, dim=1, index=topk_idx)
            
            gains = (2.0 ** topk_rel.float()) - 1.0
            positions = torch.arange(k, device=self.device)
            discounts = 1.0 / torch.log2(positions + 2)
            dcg = (gains * discounts).sum(dim=1)
            
            sorted_rel, _ = relevance.sort(dim=-1, descending=True)
            ideal_gains = (2.0 ** sorted_rel[:, :k].float()) - 1.0
            idcg = (ideal_gains * discounts).sum(dim=1).clamp(min=1e-10)
            
            ndcg = dcg / idcg
            all_ndcg.extend(ndcg.tolist())
            
        return {
            f"{split}/ndcg@{cfg.ndcg_k}": sum(all_ndcg) / len(all_ndcg),
            f"{split}/ndcg_std": torch.tensor(all_ndcg).std().item()
        }
    
    def train(self):
        cfg = self.config
        
        logger.info("Starting training...")
        pbar = tqdm(range(cfg.num_steps), desc="Training")
        
        for self.step in pbar:
            train_metrics = self.train_step()
            
            if self.step % cfg.log_interval == 0:
                pbar.set_postfix({
                    "ndcg": f"{train_metrics['train/ndcg_mean']:.4f}",
                    "sigma": f"{train_metrics['train/sigma']:.4f}"
                })
                
                if cfg.use_wandb and WANDB_AVAILABLE:
                    wandb.log(train_metrics, step=self.step)
            
            if self.step % cfg.eval_interval == 0:
                val_metrics = self.evaluate("val")
                train_eval_metrics = self.evaluate("train")
                
                logger.info(
                    f"Step {self.step}: "
                    f"Train NDCG={train_eval_metrics['train/ndcg@' + str(cfg.ndcg_k)]:.4f}, "
                    f"Val NDCG={val_metrics['val/ndcg@' + str(cfg.ndcg_k)]:.4f}"
                )
                
                if cfg.use_wandb and WANDB_AVAILABLE:
                    wandb.log({**val_metrics, **train_eval_metrics}, step=self.step)
                    
            if self.step % 1000 == 0 and self.step > 0:
                self.save_checkpoint()
                
        final_metrics = self.evaluate("val")
        logger.info(f"Final Val NDCG@{cfg.ndcg_k}: {final_metrics['val/ndcg@' + str(cfg.ndcg_k)]:.4f}")
        
        self.save_checkpoint(final=True)
        
        if cfg.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
            
        return final_metrics
    
    def save_checkpoint(self, final: bool = False):
        cfg = self.config
        path = Path(cfg.checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        filename = "final.pt" if final else f"step_{self.step}.pt"
        
        torch.save({
            "step": self.step,
            "W": self.W,
            "config": vars(cfg),
            "sigma": self.sigma_adapter.sigma if self.sigma_adapter else cfg.sigma
        }, path / filename)
        
        logger.info(f"Saved checkpoint: {path / filename}")


def main():
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="NanoMSMARCO")
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--population_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    
    config = TrainConfig(
        task=args.task,
        num_steps=args.num_steps,
        population_size=args.population_size,
        learning_rate=args.learning_rate,
        sigma=args.sigma,
        seed=args.seed,
        device=args.device,
        use_wandb=not args.no_wandb
    )
    
    trainer = EGGROLLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
