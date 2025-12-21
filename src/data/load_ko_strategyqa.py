"""
Ko-StrategyQA Dataset Loader

Korean Information Retrieval dataset from taeminlee/Ko-StrategyQA.
Format similar to NanoBEIR: queries, corpus, qrels.
"""

from datasets import load_dataset
from typing import Dict, List, Tuple
import hashlib


class KoStrategyQALoader:
    DATASET_ID = "mteb/Ko-StrategyQA"
    
    def __init__(self, seed: int = 42, max_corpus: int = 5000):
        self.seed = seed
        self.max_corpus = max_corpus
        
    def load_all(self) -> Dict:
        queries_ds = load_dataset(self.DATASET_ID, "queries", split="dev")
        corpus_ds = load_dataset(self.DATASET_ID, "corpus", split="dev")
        qrels_ds = load_dataset(self.DATASET_ID, "qrels", split="dev")
        
        queries = {row["_id"]: row["text"] for row in queries_ds}
        
        corpus = {}
        for row in corpus_ds:
            doc_id = row["_id"]
            title = row.get("title", "")
            text = row.get("text", "")
            if title:
                corpus[doc_id] = f"{title} {text}".strip()
            else:
                corpus[doc_id] = text
                
            if len(corpus) >= self.max_corpus:
                break
        
        qrels = {}
        for row in qrels_ds:
            qid = row["query-id"]
            did = row["corpus-id"]
            score = row.get("score", 1)
            if score > 0:
                if qid not in qrels:
                    qrels[qid] = []
                if did in corpus:
                    qrels[qid].append(did)
        
        valid_qids = [qid for qid in queries.keys() if qid in qrels and len(qrels[qid]) > 0]
        queries = {qid: queries[qid] for qid in valid_qids}
        qrels = {qid: qrels[qid] for qid in valid_qids}
        
        return {
            "queries": queries,
            "corpus": corpus,
            "qrels": qrels,
            "bm25": {}
        }
    
    def split_queries(
        self, 
        query_ids: List[str], 
        train_ratio: float = 0.8
    ) -> Tuple[List[str], List[str]]:
        def hash_id(qid: str) -> float:
            h = hashlib.md5(f"{qid}_{self.seed}".encode()).hexdigest()
            return int(h, 16) / (2**128)
        
        train_ids = [qid for qid in query_ids if hash_id(qid) < train_ratio]
        val_ids = [qid for qid in query_ids if hash_id(qid) >= train_ratio]
        
        return train_ids, val_ids
