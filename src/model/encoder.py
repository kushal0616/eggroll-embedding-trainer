import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List


class ModernBERTEncoder(nn.Module):
    MODEL_NAME = "answerdotai/ModernBERT-base"
    
    def __init__(self, pooling: str = "mean", max_length: int = 512):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.pooling = pooling
        self.max_length = max_length
        self.hidden_size = self.backbone.config.hidden_size
        
    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1e-9)
            return summed / lengths
        elif self.pooling == "cls":
            return hidden[:, 0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
            
    def encode(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.backbone.device)
        
        return self.forward(inputs.input_ids, inputs.attention_mask)
    
    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
