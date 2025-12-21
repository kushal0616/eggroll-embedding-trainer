"""
NanoBEIR Dataset Loader

Dataset: sentence-transformers/NanoBEIR-en
Subsets: queries, corpus, qrels, bm25
Splits: NanoMSMARCO, NanoNQ, NanoClimateFEVER, etc.
"""

from datasets import load_dataset
from typing import Dict, List, Tuple
import hashlib


class NanoBEIRLoader:
    DATASET_ID = "sentence-transformers/NanoBEIR-en"
    
    TASKS = [
        "NanoClimateFEVER", "NanoDBPedia", "NanoFEVER", 
        "NanoFiQA2018", "NanoHotpotQA", "NanoMSMARCO",
        "NanoNFCorpus", "NanoNQ", "NanoQuoraRetrieval",
        "NanoSCIDOCS", "NanoArguAna", "NanoSciFact", "NanoTouche2020"
    ]
    
    def __init__(self, task: str, seed: int = 42):
        if task not in self.TASKS:
            raise ValueError(f"Unknown task: {task}. Available: {self.TASKS}")
        self.task = task
        self.seed = seed
        
    def load_all(self) -> Dict:
        """Load queries, corpus, qrels, bm25 for a task"""
        queries = load_dataset(self.DATASET_ID, "queries", split=self.task)
        corpus = load_dataset(self.DATASET_ID, "corpus", split=self.task)
        qrels = load_dataset(self.DATASET_ID, "qrels", split=self.task)
        bm25 = load_dataset(self.DATASET_ID, "bm25", split=self.task)
        
        return {
            "queries": self._to_dict(queries, "_id", "text"),
            "corpus": self._to_corpus_dict(corpus),
            "qrels": self._build_qrels(qrels),
            "bm25": self._build_bm25(bm25)
        }
    
    def _to_dict(self, dataset, id_col: str, text_col: str) -> Dict[str, str]:
        """Convert dataset to {id: text} dict"""
        return {row[id_col]: row[text_col] for row in dataset}
    
    def _to_corpus_dict(self, corpus) -> Dict[str, str]:
        """Convert corpus dataset to {id: text} dict, combining title and text"""
        result = {}
        for row in corpus:
            doc_id = row["_id"]
            title = row.get("title", "")
            text = row.get("text", "")
            # Combine title and text
            if title:
                result[doc_id] = f"{title} {text}".strip()
            else:
                result[doc_id] = text
        return result
    
    def _build_qrels(self, qrels) -> Dict[str, List[str]]:
        """Build query -> [positive_doc_ids] mapping"""
        result = {}
        for row in qrels:
            qid = row["query-id"]
            did = row["corpus-id"]
            score = row.get("score", 1)
            if score > 0:  # Only include positive relevance
                if qid not in result:
                    result[qid] = []
                result[qid].append(did)
        return result
    
    def _build_bm25(self, bm25) -> Dict[str, List[str]]:
        """Build query -> [ranked_doc_ids] mapping from BM25 results"""
        result = {}
        for row in bm25:
            qid = row["query-id"]
            # corpus-ids is already a ranked list
            result[qid] = row["corpus-ids"]
        return result
    
    def split_queries(
        self, 
        query_ids: List[str], 
        train_ratio: float = 0.8
    ) -> Tuple[List[str], List[str]]:
        """
        Deterministic train/val split based on hash
        ~40 train, ~10 val for typical 50-query tasks
        """
        def hash_id(qid: str) -> float:
            h = hashlib.md5(f"{qid}_{self.seed}".encode()).hexdigest()
            return int(h, 16) / (2**128)
        
        train_ids = [qid for qid in query_ids if hash_id(qid) < train_ratio]
        val_ids = [qid for qid in query_ids if hash_id(qid) >= train_ratio]
        
        return train_ids, val_ids
