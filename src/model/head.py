import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Projection head W for EGGROLL. Maps [input_size] -> [output_size].
    For small datasets, use output_size=256 to reduce parameters.
    """
    
    def __init__(
        self, 
        input_size: int = 768, 
        output_size: int = 256,
        normalize: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.normalize = normalize
        
        if output_size <= input_size:
            init_W = torch.zeros(output_size, input_size)
            init_W[:output_size, :output_size] = torch.eye(output_size)
        else:
            init_W = torch.eye(output_size, input_size)
        
        self.W = nn.Parameter(init_W)
        
    def forward(self, H: torch.Tensor) -> torch.Tensor:
        projected = H @ self.W.T
        
        if self.normalize:
            projected = F.normalize(projected, p=2, dim=-1)
            
        return projected
    
    def get_flat_params(self) -> torch.Tensor:
        return self.W.view(-1)
    
    def set_flat_params(self, flat: torch.Tensor):
        self.W.data = flat.view(self.output_size, self.input_size)
