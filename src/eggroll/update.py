"""
EGGROLL Weight Update

Update rule: W <- W + alpha * (1/M) * sum(delta_f_j * (a_j @ b_j^T))
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class UpdateConfig:
    learning_rate: float = 0.05
    clip_norm: float = 1.0
    weight_decay: float = 1e-4
    momentum: float = 0.0
    use_ema: bool = False
    ema_decay: float = 0.999


class EGGROLLUpdater:
    def __init__(
        self, 
        config: UpdateConfig, 
        output_size: int,
        input_size: int = None,
        device: str = "cuda"
    ):
        self.config = config
        self.output_size = output_size
        self.input_size = input_size if input_size is not None else output_size
        self.device = device
        
        if config.momentum > 0:
            self.velocity = torch.zeros(output_size, self.input_size, device=device)
        else:
            self.velocity = None
            
        self.ema_W: Optional[torch.Tensor] = None
        
    def compute_update(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        shaped_fitness: torch.Tensor,
        sigma: float
    ) -> torch.Tensor:
        """
        Compute delta_W = (1/M) * sum(f_j * (a_j @ b_j^T))
        Efficient: (A * f.unsqueeze(-1))^T @ B
        """
        M = A.shape[0]
        weighted_A = A * shaped_fitness.unsqueeze(-1)
        delta_W = weighted_A.T @ B
        return delta_W / M
    
    def clip_update(self, delta_W: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(delta_W, p='fro')
        if norm > self.config.clip_norm:
            delta_W = delta_W * (self.config.clip_norm / norm)
        return delta_W
    
    def apply_update(
        self,
        W: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        shaped_fitness: torch.Tensor,
        sigma: float
    ) -> torch.Tensor:
        cfg = self.config
        
        delta_W = self.compute_update(A, B, shaped_fitness, sigma)
        delta_W = self.clip_update(delta_W)
        
        if self.velocity is not None:
            self.velocity = cfg.momentum * self.velocity + delta_W
            delta_W = self.velocity
        
        W_new = W + cfg.learning_rate * delta_W
        
        if cfg.weight_decay > 0:
            W_new = W_new * (1 - cfg.learning_rate * cfg.weight_decay)
        
        if cfg.use_ema:
            if self.ema_W is None:
                self.ema_W = W_new.clone()
            else:
                self.ema_W = cfg.ema_decay * self.ema_W + (1 - cfg.ema_decay) * W_new
        
        return W_new
    
    def get_eval_weights(self, W: torch.Tensor) -> torch.Tensor:
        if self.config.use_ema and self.ema_W is not None:
            return self.ema_W
        return W


class AdaptiveSigmaUpdater:
    """Adapts sigma based on fitness variance."""
    
    def __init__(
        self,
        initial_sigma: float = 0.02,
        min_sigma: float = 0.005,
        max_sigma: float = 0.1,
        target_variance: float = 0.1,
        adaptation_rate: float = 0.01
    ):
        self.sigma = initial_sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.target_variance = target_variance
        self.adaptation_rate = adaptation_rate
        
    def update(self, fitness_pos: torch.Tensor, fitness_neg: torch.Tensor) -> float:
        diff = fitness_pos - fitness_neg
        variance = diff.var().item()
        
        if variance < self.target_variance * 0.5:
            self.sigma = min(self.sigma * (1 + self.adaptation_rate), self.max_sigma)
        elif variance > self.target_variance * 2.0:
            self.sigma = max(self.sigma * (1 - self.adaptation_rate), self.min_sigma)
            
        return self.sigma
