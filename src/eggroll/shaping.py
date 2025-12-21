"""
Fitness Shaping for EGGROLL

Normalizes fitness values to stabilize gradient estimates.
"""

import torch
from typing import Literal, Optional


class FitnessShaper:
    def __init__(self, method: Literal["rank", "zscore", "combined"] = "rank"):
        self.method = method
        
    def rank_transform(self, fitness: torch.Tensor) -> torch.Tensor:
        """Transform fitness to ranks in [-0.5, 0.5]."""
        N = fitness.shape[0]
        _, indices = fitness.sort()
        ranks = torch.zeros_like(fitness)
        ranks[indices] = torch.arange(N, dtype=fitness.dtype, device=fitness.device)
        return (ranks / (N - 1)) - 0.5
    
    def zscore(self, fitness: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Z-score normalization."""
        mean = fitness.mean(dim=-1, keepdim=True)
        std = fitness.std(dim=-1, keepdim=True).clamp(min=eps)
        return (fitness - mean) / std
    
    def per_query_zscore(self, fitness: torch.Tensor) -> torch.Tensor:
        """Z-score per query, then average across queries."""
        shaped = self.zscore(fitness)
        return shaped.mean(dim=0)
    
    def __call__(
        self,
        fitness: torch.Tensor,
        per_query_fitness: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.method == "rank":
            return self.rank_transform(fitness)
        elif self.method == "zscore":
            if per_query_fitness is not None:
                return self.per_query_zscore(per_query_fitness)
            return self.zscore(fitness)
        elif self.method == "combined":
            if per_query_fitness is not None:
                zscored = self.per_query_zscore(per_query_fitness)
            else:
                zscored = self.zscore(fitness)
            return self.rank_transform(zscored)
        else:
            raise ValueError(f"Unknown method: {self.method}")


class AntitheticShaper:
    """Computes fitness difference for antithetic pairs."""
    
    def __call__(self, fitness: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fitness: [N] where N = 2M, first M positive, last M negative
        Returns:
            delta: [M] fitness difference (f+ - f-) / 2
        """
        M = fitness.shape[0] // 2
        return (fitness[:M] - fitness[M:]) / 2
