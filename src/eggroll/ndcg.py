"""
GPU-Accelerated NDCG@k Computation

Uses topk instead of full sort for efficiency.
Caches IDCG per query since it only depends on relevance distribution.
"""

import torch
from typing import Optional, Tuple


class NDCGComputer:
    def __init__(self, k: int = 20, device: str = "cuda"):
        self.k = k
        self.device = device
        
        positions = torch.arange(k, device=device, dtype=torch.float32)
        self.discounts = 1.0 / torch.log2(positions + 2)
        
    def compute_dcg(self, relevance: torch.Tensor) -> torch.Tensor:
        """DCG@k = sum((2^rel - 1) / log2(i + 2)) for top-k items."""
        gains = (2.0 ** relevance) - 1.0
        return (gains * self.discounts[:relevance.shape[-1]]).sum(dim=-1)
    
    def compute_idcg(self, relevance: torch.Tensor) -> torch.Tensor:
        """Ideal DCG: best possible DCG for given relevance distribution."""
        sorted_rel, _ = relevance.sort(dim=-1, descending=True)
        ideal_topk = sorted_rel[..., :self.k]
        return self.compute_dcg(ideal_topk.float())
    
    def compute_ndcg(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        idcg: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute NDCG@k for all perturbations.
        
        Args:
            scores: [B, P, N] scores for each (query, doc, perturbation)
            relevance: [B, P] ground truth relevance
            idcg: [B] optional precomputed ideal DCG
            
        Returns:
            ndcg_mean: [N] NDCG averaged over queries
            ndcg_per_query: [B, N] per-query NDCG
        """
        B, P, N = scores.shape
        k = min(self.k, P)
        
        _, topk_indices = scores.topk(k, dim=1)
        
        rel_expanded = relevance.unsqueeze(-1).expand(-1, -1, N)
        topk_rel = torch.gather(rel_expanded, dim=1, index=topk_indices)
        
        gains = (2.0 ** topk_rel.float()) - 1.0
        dcg = (gains * self.discounts[:k].view(1, k, 1)).sum(dim=1)
        
        if idcg is None:
            idcg = self.compute_idcg(relevance)
            
        ndcg = dcg / idcg.unsqueeze(-1).clamp(min=1e-10)
        ndcg_mean = ndcg.mean(dim=0)
        
        return ndcg_mean, ndcg


class CachedNDCGComputer(NDCGComputer):
    """Version with IDCG caching for repeated evaluation on same pools."""
    
    def __init__(self, k: int = 20, device: str = "cuda"):
        super().__init__(k, device)
        self.idcg_cache = {}
        
    def cache_idcg(self, query_ids: list, relevance: torch.Tensor):
        idcg = self.compute_idcg(relevance)
        for i, qid in enumerate(query_ids):
            self.idcg_cache[qid] = idcg[i].item()
            
    def get_cached_idcg(self, query_ids: list) -> torch.Tensor:
        return torch.tensor(
            [self.idcg_cache[qid] for qid in query_ids],
            device=self.device
        )
