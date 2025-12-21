"""
Candidate Pool Builder

Pool structure per query:
- Positives (rel=2): from qrels, capped at pos_cap
- Hard negatives (rel=0): from BM25 top-K, excluding positives
- Random negatives (rel=0): random sample from corpus
- Ambiguous (rel=1): optional, for graded relevance
"""

from dataclasses import dataclass
from typing import Dict, List, Set
import numpy as np


@dataclass
class PoolConfig:
    pool_size: int = 256          # P: total pool size
    pos_cap: int = 3              # Max positives per query
    rand_count: int = 16          # Random negatives
    amb_count: int = 0            # Ambiguous candidates (Phase 2)
    bm25_depth: int = 100         # How deep to look in BM25 results
    

@dataclass 
class QueryPool:
    query_id: str
    doc_ids: List[str]            # Ordered list of doc IDs
    relevance: np.ndarray         # rel[i] in {0, 1, 2} for doc_ids[i]
    

class PoolBuilder:
    def __init__(self, config: PoolConfig):
        self.config = config
        
    def build_pool(
        self,
        query_id: str,
        qrels: Dict[str, List[str]],      # query_id -> [positive_doc_ids]
        bm25: Dict[str, List[str]],       # query_id -> [ranked_doc_ids]
        corpus_ids: List[str],
        rng: np.random.Generator
    ) -> QueryPool:
        """Build candidate pool for a single query"""
        cfg = self.config
        
        # 1. Positives (rel=2)
        positives = qrels.get(query_id, [])[:cfg.pos_cap]
        positive_set = set(positives)
        
        # 2. Hard negatives from BM25 (rel=0)
        bm25_candidates = bm25.get(query_id, [])[:cfg.bm25_depth]
        hard_negs = [d for d in bm25_candidates if d not in positive_set]
        
        # 3. Random negatives (rel=0)
        available = [d for d in corpus_ids if d not in positive_set]
        rand_negs = rng.choice(
            available, 
            size=min(cfg.rand_count, len(available)),
            replace=False
        ).tolist()
        
        # 4. Combine and pad/truncate to pool_size
        # FIX: Use len(rand_negs) not cfg.rand_count (rand_negs may be shorter)
        remaining_slots = max(0, cfg.pool_size - len(positives) - len(rand_negs))
        hard_negs = hard_negs[:remaining_slots]
        
        # Final pool
        doc_ids = positives + hard_negs + rand_negs
        
        # Pad if needed
        if len(doc_ids) < cfg.pool_size:
            extra_available = [d for d in corpus_ids if d not in set(doc_ids)]
            if extra_available:
                extra = rng.choice(
                    extra_available,
                    size=min(cfg.pool_size - len(doc_ids), len(extra_available)),
                    replace=False
                ).tolist()
                doc_ids.extend(extra)
        
        # Truncate if needed
        doc_ids = doc_ids[:cfg.pool_size]
        
        # Build relevance array
        relevance = np.zeros(len(doc_ids), dtype=np.int32)
        for i, did in enumerate(doc_ids):
            if did in positive_set:
                relevance[i] = 2
                
        return QueryPool(query_id, doc_ids, relevance)
    
    def build_all_pools(
        self,
        query_ids: List[str],
        qrels: Dict,
        bm25: Dict,
        corpus_ids: List[str],
        seed: int = 42
    ) -> Dict[str, QueryPool]:
        """Build pools for all queries"""
        rng = np.random.default_rng(seed)
        return {
            qid: self.build_pool(qid, qrels, bm25, corpus_ids, rng)
            for qid in query_ids
        }
