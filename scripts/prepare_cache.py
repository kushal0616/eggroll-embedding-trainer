#!/usr/bin/env python
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

import torch
from tqdm import tqdm

from src.data.load_nanobeir import NanoBEIRLoader
from src.data.cache_prehead import EmbeddingCache
from src.model.encoder import ModernBERTEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_cache(
    tasks: list,
    cache_dir: str = "./cache",
    device: str = "cuda",
    batch_size: int = 64
):
    cache = EmbeddingCache(Path(cache_dir), device=device)
    
    logger.info("Loading ModernBERT encoder...")
    encoder = ModernBERTEncoder(pooling="mean")
    encoder.to(device)
    encoder.freeze()
    encoder.eval()
    
    for task in tqdm(tasks, desc="Tasks"):
        logger.info(f"Processing task: {task}")
        
        query_cache_file = Path(cache_dir) / f"{task}_queries_embeddings.pt"
        corpus_cache_file = Path(cache_dir) / f"{task}_corpus_embeddings.pt"
        
        if query_cache_file.exists() and corpus_cache_file.exists():
            logger.info(f"  Cache already exists for {task}, skipping...")
            continue
        
        loader = NanoBEIRLoader(task)
        data = loader.load_all()
        
        logger.info(f"  Caching {len(data['queries'])} queries...")
        cache.compute_and_save(
            encoder, 
            data["queries"], 
            batch_size=batch_size,
            task_name=f"{task}_queries"
        )
        
        logger.info(f"  Caching {len(data['corpus'])} documents...")
        cache.compute_and_save(
            encoder,
            data["corpus"],
            batch_size=batch_size,
            task_name=f"{task}_corpus"
        )
        
        logger.info(f"  Done with {task}")
    
    logger.info("All caching complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["NanoMSMARCO"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    
    args = parser.parse_args()
    
    if args.all:
        tasks = NanoBEIRLoader.TASKS
    else:
        tasks = args.tasks
        
    prepare_cache(
        tasks=tasks,
        cache_dir=args.cache_dir,
        device=args.device,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
