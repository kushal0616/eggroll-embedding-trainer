#!/usr/bin/env python
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

from src.train import EGGROLLTrainer, TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="NanoMSMARCO")
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--population_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--no_wandb", action="store_true")
    
    args = parser.parse_args()
    
    config = TrainConfig(
        task=args.task,
        num_steps=args.num_steps,
        population_size=args.population_size,
        learning_rate=args.learning_rate,
        sigma=args.sigma,
        seed=args.seed,
        device=args.device,
        cache_dir=args.cache_dir,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=not args.no_wandb
    )
    
    logger.info(f"Starting experiment: {args.task}")
    trainer = EGGROLLTrainer(config)
    results = trainer.train()
    
    logger.info(f"Final results: {results}")


if __name__ == "__main__":
    main()
