import torch
from dataclasses import dataclass
from typing import Tuple


@dataclass
class NoiseConfig:
    rank: int = 1
    population_size: int = 256
    sigma: float = 0.02
    seed: int = 42


class Rank1NoiseGenerator:
    """
    Generates rank-1 perturbations for EGGROLL: E = a @ b^T
    Uses antithetic sampling: for each (a,b), uses both +sigma*E and -sigma*E
    
    For W: [D_out, D_in], generates A: [M, D_out] and B: [M, D_in]
    """
    
    def __init__(
        self, 
        config: NoiseConfig, 
        output_size: int,
        input_size: int = None,
        device: str = "cuda"
    ):
        self.config = config
        self.output_size = output_size
        self.input_size = input_size if input_size is not None else output_size
        self.device = device
        
        assert config.population_size % 2 == 0, "Population must be even for antithetic sampling"
        self.num_directions = config.population_size // 2
        
    def sample(self, step: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns A: [M, D_out], B: [M, D_in] where M = population_size // 2"""
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config.seed + step)
        
        M = self.num_directions
        
        A = torch.randn(M, self.output_size, generator=generator, device=self.device)
        B = torch.randn(M, self.input_size, generator=generator, device=self.device)
        
        return A, B
    
    def get_antithetic_signs(self) -> torch.Tensor:
        """Returns [+1, +1, ..., -1, -1, ...] of shape [N] where N = population_size"""
        M = self.num_directions
        return torch.cat([
            torch.ones(M, device=self.device),
            -torch.ones(M, device=self.device)
        ])
