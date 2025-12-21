"""
Vectorized Score Computation for EGGROLL

Key insight: compute all N perturbed scores in one pass without materializing E.

For rank-1 perturbation W_i = W + sigma * a_i @ b_i^T:
    score_i = (q_base + sigma*s_q*a) . (d_base + sigma*s_d*a)
    
Where s_q = H_q @ b, s_d = H_d @ b are precomputed scalar products.
"""

import torch
from typing import Tuple


class VectorizedScorer:
    def __init__(self, sigma: float = 0.02):
        self.sigma = sigma
        
    def compute_base_scores(
        self,
        H_q: torch.Tensor,
        H_d: torch.Tensor,
        W: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute unperturbed embeddings and scores.
        
        Args:
            H_q: [B, D_in] query embeddings
            H_d: [B, P, D_in] document embeddings
            W: [D_out, D_in] projection matrix
            
        Returns:
            q_base: [B, D_out]
            d_base: [B, P, D_out]
            scores_base: [B, P]
        """
        q_base = H_q @ W.T
        d_base = torch.einsum('bpd,ed->bpe', H_d, W)
        scores_base = torch.einsum('bd,bpd->bp', q_base, d_base)
        
        return q_base, d_base, scores_base
    
    def compute_perturbed_scores_rank1(
        self,
        H_q: torch.Tensor,
        H_d: torch.Tensor,
        q_base: torch.Tensor,
        d_base: torch.Tensor,
        scores_base: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scores for all perturbations without materializing E.
        
        Math: score_i = base + sigma*(s_q*ad + s_d*aq) + sigma^2*s_q*s_d*a2
        
        Returns: [B, P, N] where N = 2*M (antithetic pairs)
        """
        sigma = self.sigma
        M = A.shape[0]
        
        s_q = H_q @ B.T
        s_d = torch.einsum('bpd,md->bpm', H_d, B)
        
        aq = q_base @ A.T
        ad = torch.einsum('bpd,md->bpm', d_base, A)
        
        a2 = (A * A).sum(dim=-1)
        
        linear_term = s_q.unsqueeze(1) * ad + s_d * aq.unsqueeze(1)
        quad_term = s_q.unsqueeze(1) * s_d * a2.view(1, 1, M)
        
        scores_pos = scores_base.unsqueeze(-1) + sigma * linear_term + sigma**2 * quad_term
        scores_neg = scores_base.unsqueeze(-1) - sigma * linear_term + sigma**2 * quad_term
        
        return torch.cat([scores_pos, scores_neg], dim=-1)
    
    def compute_all_scores(
        self,
        H_q: torch.Tensor,
        H_d: torch.Tensor,
        W: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        """Full pipeline: returns [B, P, N] scores for all perturbations."""
        q_base, d_base, scores_base = self.compute_base_scores(H_q, H_d, W)
        return self.compute_perturbed_scores_rank1(
            H_q, H_d, q_base, d_base, scores_base, A, B
        )


class ChunkedScorer(VectorizedScorer):
    """Memory-efficient version that chunks over population dimension."""
    
    def __init__(self, sigma: float = 0.02, chunk_size: int = 64):
        super().__init__(sigma)
        self.chunk_size = chunk_size
        
    def compute_all_scores_chunked(
        self,
        H_q: torch.Tensor,
        H_d: torch.Tensor,
        W: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        q_base, d_base, scores_base = self.compute_base_scores(H_q, H_d, W)
        
        M = A.shape[0]
        all_scores = []
        
        for start in range(0, M, self.chunk_size):
            end = min(start + self.chunk_size, M)
            scores_chunk = self.compute_perturbed_scores_rank1(
                H_q, H_d, q_base, d_base, scores_base,
                A[start:end], B[start:end]
            )
            all_scores.append(scores_chunk)
            
        return torch.cat(all_scores, dim=-1)
