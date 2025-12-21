"""
Baseline: Gradient-based Contrastive Learning Trainer

Uses InfoNCE/contrastive loss which is differentiable, unlike NDCG.
This serves as a comparison baseline for EGGROLL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
import logging
from tqdm import tqdm

from src.data.load_nanobeir import NanoBEIRLoader
from src.data.build_pools import PoolBuilder, PoolConfig, QueryPool
from src.data.cache_prehead import EmbeddingCache

logger = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    task: str = "NanoMSMARCO"
    train_ratio: float = 0.8
    pool_size: int = 64
    pos_cap: int = 3
    rand_count: int = 8
    hidden_size: int = 768
    head_output_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.05
    num_steps: int = 500
    eval_interval: int = 100
    ndcg_k: int = 10
    device: str = "mps"
    seed: int = 42
    cache_dir: str = "./cache"


class ProjectionHead(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        return F.normalize(out, dim=-1)


class ContrastiveTrainer:
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.device = config.device
        self.step = 0
        
        torch.manual_seed(config.seed)
        
        self._init_data()
        self._init_model()
        
    def _init_data(self):
        cfg = self.config
        
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
        
        cache = EmbeddingCache(Path(cfg.cache_dir), device=self.device)
        self.query_cache = cache.load(f"{cfg.task}_queries")
        self.doc_cache = cache.load(f"{cfg.task}_corpus")
        
    def _init_model(self):
        cfg = self.config
        
        self.head = ProjectionHead(cfg.hidden_size, cfg.head_output_size)
        self.head.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
        
    def get_batch(self, query_ids: List[str], pools: Dict[str, QueryPool]):
        H_q = self.query_cache.get_embeddings(query_ids)
        
        doc_ids_batch = [pools[qid].doc_ids for qid in query_ids]
        H_d = torch.stack([
            self.doc_cache.get_embeddings(doc_ids)
            for doc_ids in doc_ids_batch
        ])
        
        relevance = torch.stack([
            torch.tensor(pools[qid].relevance, device=self.device)
            for qid in query_ids
        ])
        
        return H_q, H_d, relevance
    
    def contrastive_loss(
        self, 
        q_emb: torch.Tensor,
        d_emb: torch.Tensor, 
        relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE-style contrastive loss.
        
        Args:
            q_emb: [B, D] normalized query embeddings
            d_emb: [B, P, D] normalized document embeddings
            relevance: [B, P] relevance labels (0, 1, 2)
        """
        cfg = self.config
        
        scores = torch.einsum('bd,bpd->bp', q_emb, d_emb) / cfg.temperature
        
        pos_mask = (relevance > 0).float()
        
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        
        log_softmax = F.log_softmax(scores, dim=-1)
        
        loss_per_query = -(log_softmax * pos_mask).sum(dim=-1) / pos_mask.sum(dim=-1).clamp(min=1)
        
        return loss_per_query.mean()
    
    def train_step(self) -> Dict[str, float]:
        self.head.train()
        
        H_q, H_d, relevance = self.get_batch(self.train_query_ids, self.train_pools)
        
        q_emb = self.head(H_q)
        d_emb = self.head(H_d.view(-1, H_d.shape[-1])).view(H_d.shape[0], H_d.shape[1], -1)
        
        loss = self.contrastive_loss(q_emb, d_emb, relevance)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
        self.optimizer.step()
        
        return {"train/loss": loss.item()}
    
    @torch.no_grad()
    def compute_ndcg(
        self, 
        query_ids: List[str], 
        pools: Dict[str, QueryPool]
    ) -> float:
        self.head.eval()
        cfg = self.config
        
        if not query_ids:
            return 0.0
            
        H_q, H_d, relevance = self.get_batch(query_ids, pools)
        
        q_emb = self.head(H_q)
        d_emb = self.head(H_d.view(-1, H_d.shape[-1])).view(H_d.shape[0], H_d.shape[1], -1)
        
        scores = torch.einsum('bd,bpd->bp', q_emb, d_emb)
        
        k = min(cfg.ndcg_k, scores.shape[1])
        _, topk_idx = scores.topk(k, dim=1)
        topk_rel = torch.gather(relevance, dim=1, index=topk_idx)
        
        positions = torch.arange(k, device=self.device)
        discounts = 1.0 / torch.log2(positions + 2)
        
        gains = (2.0 ** topk_rel.float()) - 1.0
        dcg = (gains * discounts).sum(dim=1)
        
        sorted_rel, _ = relevance.sort(dim=-1, descending=True)
        ideal_gains = (2.0 ** sorted_rel[:, :k].float()) - 1.0
        idcg = (ideal_gains * discounts).sum(dim=1).clamp(min=1e-10)
        
        ndcg = dcg / idcg
        return ndcg.mean().item()
    
    def train(self) -> Dict[str, float]:
        cfg = self.config
        
        logger.info(f"Baseline training: {cfg.num_steps} steps, lr={cfg.learning_rate}")
        
        best_val = 0.0
        history = {"train_ndcg": [], "val_ndcg": [], "loss": []}
        
        pbar = tqdm(range(cfg.num_steps), desc="Baseline Training")
        
        for self.step in pbar:
            metrics = self.train_step()
            history["loss"].append(metrics["train/loss"])
            
            if self.step % cfg.eval_interval == 0:
                train_ndcg = self.compute_ndcg(self.train_query_ids, self.train_pools)
                val_ndcg = self.compute_ndcg(self.val_query_ids, self.val_pools)
                
                history["train_ndcg"].append(train_ndcg)
                history["val_ndcg"].append(val_ndcg)
                
                best_val = max(best_val, val_ndcg)
                
                pbar.set_postfix({
                    "loss": f"{metrics['train/loss']:.4f}",
                    "train": f"{train_ndcg:.4f}",
                    "val": f"{val_ndcg:.4f}"
                })
                
                logger.info(
                    f"Step {self.step}: Loss={metrics['train/loss']:.4f}, "
                    f"Train NDCG={train_ndcg:.4f}, Val NDCG={val_ndcg:.4f}"
                )
        
        final_train = self.compute_ndcg(self.train_query_ids, self.train_pools)
        final_val = self.compute_ndcg(self.val_query_ids, self.val_pools)
        
        return {
            "final_train_ndcg": final_train,
            "final_val_ndcg": final_val,
            "best_val_ndcg": best_val,
            "history": history
        }


def run_baseline(
    task: str = "NanoMSMARCO",
    num_steps: int = 500,
    learning_rate: float = 1e-3,
    device: str = "mps"
) -> Dict:
    """Run baseline contrastive training."""
    
    config = BaselineConfig(
        task=task,
        num_steps=num_steps,
        learning_rate=learning_rate,
        device=device
    )
    
    trainer = ContrastiveTrainer(config)
    return trainer.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_baseline()
    print(f"\nBaseline Results: {results}")
