import torch
import numpy as np
from typing import Dict, List
from pathlib import Path
import argparse
import logging

from src.data.load_nanobeir import NanoBEIRLoader
from src.data.cache_prehead import EmbeddingCache

logger = logging.getLogger(__name__)


class FullCorpusEvaluator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        
    def evaluate(
        self,
        W: torch.Tensor,
        query_cache,
        doc_cache,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        qrels: Dict[str, List[str]],
        k_values: List[int] = [10, 20, 100]
    ) -> Dict[str, float]:
        results = {}
        
        query_ids = list(queries.keys())
        doc_ids = list(corpus.keys())
        
        Q = query_cache.get_embeddings(query_ids) @ W.T
        D = doc_cache.get_embeddings(doc_ids) @ W.T
        
        max_k = max(k_values)
        batch_size = 32
        
        all_rankings = {}
        
        for i in range(0, len(query_ids), batch_size):
            batch_q = Q[i:i+batch_size]
            scores = batch_q @ D.T
            
            _, topk_indices = scores.topk(min(max_k, len(doc_ids)), dim=-1)
            
            for j, qid in enumerate(query_ids[i:i+batch_size]):
                all_rankings[qid] = [doc_ids[idx] for idx in topk_indices[j].tolist()]
        
        for k in k_values:
            ndcg_scores = []
            recall_scores = []
            mrr_scores = []
            
            for qid in query_ids:
                relevant = set(qrels.get(qid, []))
                if not relevant:
                    continue
                    
                ranking = all_rankings[qid][:k]
                
                dcg = 0.0
                for rank, did in enumerate(ranking):
                    if did in relevant:
                        dcg += 1.0 / np.log2(rank + 2)
                
                ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
                ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
                ndcg_scores.append(ndcg)
                
                hits = len(set(ranking) & relevant)
                recall = hits / len(relevant)
                recall_scores.append(recall)
                
                mrr = 0.0
                for rank, did in enumerate(ranking):
                    if did in relevant:
                        mrr = 1.0 / (rank + 1)
                        break
                mrr_scores.append(mrr)
            
            if ndcg_scores:
                results[f"ndcg@{k}"] = np.mean(ndcg_scores)
                results[f"recall@{k}"] = np.mean(recall_scores)
                results[f"mrr@{k}"] = np.mean(mrr_scores)
        
        return results


def main():
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default="NanoMSMARCO")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    args = parser.parse_args()
    
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    W = checkpoint["W"].to(args.device)
    
    loader = NanoBEIRLoader(args.task)
    data = loader.load_all()
    
    cache = EmbeddingCache(Path(args.cache_dir), device=args.device)
    query_cache = cache.load(f"{args.task}_queries")
    doc_cache = cache.load(f"{args.task}_corpus")
    
    evaluator = FullCorpusEvaluator(device=args.device)
    results = evaluator.evaluate(
        W, query_cache, doc_cache,
        data["queries"], data["corpus"], data["qrels"]
    )
    
    print("\nFull Corpus Evaluation Results:")
    for metric, value in sorted(results.items()):
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
