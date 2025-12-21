import numpy as np
from typing import Dict, List


class BM25Evaluator:
    def __init__(self, k_values: List[int] = [10, 20, 100]):
        self.k_values = k_values
        
    def evaluate(
        self,
        query_ids: List[str],
        qrels: Dict[str, List[str]],
        bm25_rankings: Dict[str, List[str]]
    ) -> Dict[str, float]:
        results = {}
        
        for k in self.k_values:
            ndcg_scores = []
            recall_scores = []
            mrr_scores = []
            
            for qid in query_ids:
                relevant = set(qrels.get(qid, []))
                if not relevant:
                    continue
                    
                ranking = bm25_rankings.get(qid, [])[:k]
                
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
                
            results[f"bm25_ndcg@{k}"] = np.mean(ndcg_scores) if ndcg_scores else 0.0
            results[f"bm25_recall@{k}"] = np.mean(recall_scores) if recall_scores else 0.0
            results[f"bm25_mrr@{k}"] = np.mean(mrr_scores) if mrr_scores else 0.0
            
        return results
