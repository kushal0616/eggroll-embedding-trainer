"""
Pre-head Embedding Cache

Since backbone is frozen, we can pre-compute all embeddings once.
Storage: ~56k docs * 768 dims * 2 bytes (fp16) ~ 86MB per task
"""

import torch
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


class EmbeddingCache:
    def __init__(self, cache_dir: Path, device: str = "cuda"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
    def compute_and_save(
        self,
        encoder,
        texts: Dict[str, str],          # id -> text
        batch_size: int = 64,
        task_name: str = "default"
    ):
        """Compute embeddings for all texts and save"""
        encoder.eval()
        
        ids = list(texts.keys())
        all_texts = [texts[i] for i in ids]
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(all_texts), batch_size), desc=f"Caching {task_name}"):
                batch = all_texts[i:i+batch_size]
                emb = encoder.encode(batch)  # [B, D]
                embeddings.append(emb.cpu())
                
        embeddings = torch.cat(embeddings, dim=0).half()  # fp16
        
        # Save
        torch.save({
            "ids": ids,
            "embeddings": embeddings,
            "id_to_idx": {id_: idx for idx, id_ in enumerate(ids)}
        }, self.cache_dir / f"{task_name}_embeddings.pt")
        
    def load(self, task_name: str) -> "LoadedCache":
        """Load cached embeddings"""
        data = torch.load(self.cache_dir / f"{task_name}_embeddings.pt", weights_only=False)
        return LoadedCache(
            embeddings=data["embeddings"].to(self.device),
            id_to_idx=data["id_to_idx"]
        )


class LoadedCache:
    def __init__(self, embeddings: torch.Tensor, id_to_idx: Dict[str, int]):
        self.embeddings = embeddings  # [N, D]
        self.id_to_idx = id_to_idx
        
    def get_embeddings(self, ids: List[str]) -> torch.Tensor:
        """Get embeddings for a list of IDs"""
        indices = [self.id_to_idx[i] for i in ids]
        return self.embeddings[indices].float()  # Convert back to fp32 for computation
    
    def get_batch(
        self, 
        query_ids: List[str],
        pools: Dict[str, "QueryPool"],
        query_cache: "LoadedCache"
    ):
        """
        Get batch of query and document embeddings
        
        Returns:
            H_q: [B, D] query pre-head embeddings
            H_d: [B, P, D] document pre-head embeddings
            relevance: [B, P] relevance labels
        """
        import torch
        
        B = len(query_ids)
        P = len(pools[query_ids[0]].doc_ids)
        
        H_q = query_cache.get_embeddings(query_ids)  # [B, D]
        
        doc_ids_batch = [pools[qid].doc_ids for qid in query_ids]
        H_d = torch.stack([
            self.get_embeddings(doc_ids) 
            for doc_ids in doc_ids_batch
        ])  # [B, P, D]
        
        relevance = torch.stack([
            torch.tensor(pools[qid].relevance)
            for qid in query_ids
        ])  # [B, P]
        
        return H_q, H_d, relevance
