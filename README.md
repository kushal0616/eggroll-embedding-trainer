# EGGROLL-NDCG

A PyTorch implementation of gradient-free optimization for directly optimizing NDCG (Normalized Discounted Cumulative Gain) in neural information retrieval systems using Evolution Strategies.

---

## Original Paper: Evolution Strategies at the Hyperscale

> **Paper**: [arXiv:2511.16652](https://arxiv.org/abs/2511.16652)
>
> **Authors**: Bidipta Sarkar, Mattie Fellows, Juan Agustin Duque, Alistair Letcher, Antonio León Villares, Anya Sims, Dylan Cope, Jarek Liesen, Lukas Seier, Theo Wolf, Uljad Berdica, Alexander David Goldie, Aaron Courville, Karin Sevegnani, Shimon Whiteson, Jakob Nicolaus Foerster

### TL;DR
**"Train billion-parameter neural networks without computing a single gradient!"**

---

### The Problem with Backpropagation

Traditional deep learning relies on **backpropagation** to compute gradients. But backprop has fundamental limitations:

| Limitation | Why It Matters |
|------------|----------------|
| **Requires differentiable objectives** | Ranking metrics like NDCG, MAP, MRR are non-differentiable! |
| **Sequential computation** | Must compute forward pass, store activations, then backward pass |
| **Memory explosion** | Must store all intermediate activations for gradient computation |
| **Gradient pathologies** | Vanishing/exploding gradients, especially in long sequences |

---

### How Evolution Strategies Learn Without Backprop

This is the key insight: **you don't need gradients to find which direction improves your objective.**

#### The Core Idea: Estimate Gradients by Random Sampling

Instead of computing exact gradients through the chain rule, ES **estimates** the gradient direction by:

```
1. Add random noise to weights      →  W' = W + σ·ε
2. Evaluate fitness (forward only!) →  f(W') = NDCG score
3. Correlate noise with fitness     →  Which noise directions improved the score?
4. Update weights accordingly       →  W = W + α·(noise that helped)
```

#### Mathematical Formulation

The gradient of expected fitness can be written as:

```
∇θ E[f(θ)] = E[f(θ) · ∇θ log p(θ)]
```

For Gaussian perturbations, this simplifies to:

```
∇θ E[f(θ + σε)] ≈ (1/σ) · E[f(θ + σε) · ε]
```

**In plain English**: The gradient direction is approximately the average of "noise vectors weighted by how much they improved the objective."

#### Visual Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BACKPROPAGATION (Traditional)                         │
└─────────────────────────────────────────────────────────────────────────┘

  Input → [Layer 1] → [Layer 2] → [Layer 3] → Loss
             ↑            ↑            ↑         │
             │            │            │         │
             └────────────┴────────────┴─────────┘
                    Backward Pass (Sequential!)

  • Must store ALL activations
  • Must compute chain rule through every layer
  • Loss function MUST be differentiable


┌─────────────────────────────────────────────────────────────────────────┐
│                    EVOLUTION STRATEGIES (EGGROLL)                        │
└─────────────────────────────────────────────────────────────────────────┘

           ┌─→ W+σε₁ → Forward → f₁ ─┐
           │                          │
  W ───────┼─→ W+σε₂ → Forward → f₂ ──┼─→ Aggregate → Update W
           │                          │
           └─→ W+σε₃ → Forward → f₃ ─┘

  • Only forward passes (fully parallel!)
  • No activation storage needed
  • Loss function can be ANY black-box metric
```

#### Why This Works: Intuition

Imagine you're blindfolded on a hill, trying to find the top:

- **Gradient Descent**: Someone tells you the exact slope at your feet (but requires calculus)
- **Evolution Strategies**: You take random steps in all directions, remember which ones went uphill, then move in the average "uphill" direction

Both eventually reach the top, but ES doesn't need to know the hill's equation!

---

### EGGROLL's Innovation: Low-Rank Perturbations

The paper's key contribution is making ES **scalable** to massive neural networks.

**The Problem**: For a weight matrix W ∈ ℝ^{m×n}, generating full noise matrices costs O(m×n) memory and compute per sample.

**The Solution**: Generate **low-rank** noise instead:

```
Traditional ES:  W' = W + σ·E           where E ∈ ℝ^{m×n}  (full matrix)
EGGROLL:         W' = W + σ·(A·Bᵀ)      where A ∈ ℝ^{m×r}, B ∈ ℝ^{n×r}
```

| Metric | Traditional ES | EGGROLL (rank-r) |
|--------|---------------|------------------|
| **Noise Storage** | O(m × n) | O(r × (m + n)) |
| **Forward Cost** | O(m × n) | O(r × (m + n)) |
| **Convergence** | Baseline | O(1/r) to full-rank |

**For rank-1 (this implementation)**:
- Noise is just outer product of two vectors: `E = a ⊗ bᵀ`
- Memory: O(m + n) — **millions of times smaller!**

---

### Paper's Key Experiments

The paper validates EGGROLL across diverse domains:

| Domain | Task | Key Result |
|--------|------|------------|
| **Reinforcement Learning** | MuJoCo, Atari | Matches/exceeds PPO with better parallelization |
| **LLM Fine-tuning** | Reasoning tasks | Competitive with gradient-based LoRA |
| **Integer-only Training** | Language modeling | Enables training without floating-point! |

---

### Connection to This Implementation

This **EGGROLL-NDCG** project applies the paper's ideas to **Information Retrieval**:

| Original Paper | This Implementation |
|----------------|---------------------|
| General ES framework | IR-specialized implementation |
| Various objectives | **Direct NDCG optimization** |
| JAX-based | **PyTorch-based** |
| Billion-parameter models | Lightweight projection head |

**Why ES for IR?**
- NDCG involves **sorting and ranking** — fundamentally non-differentiable
- Traditional methods use surrogate losses (contrastive, cross-entropy) that don't directly optimize ranking
- EGGROLL treats NDCG as a **black-box fitness function** — optimize exactly what you measure!

---

## Why EGGROLL?

Traditional neural IR training faces a fundamental problem: **NDCG is non-differentiable**. The standard workaround is to use surrogate losses (contrastive, cross-entropy, etc.) that are differentiable but don't directly optimize what we care about.

EGGROLL takes a different approach: **treat NDCG as a black-box objective** and optimize it directly using Evolution Strategies. No gradients needed.

### Key Insight

| Approach | Loss Function | Optimizes NDCG? | Generalization |
|----------|---------------|-----------------|----------------|
| **Gradient-based** | Contrastive (InfoNCE) | Indirectly | Prone to overfitting |
| **EGGROLL (ES)** | NDCG directly | Yes | Better generalization |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EGGROLL Training Pipeline                          │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────────┐
                    │         Frozen Encoder               │
                    │   (ModernBERT / multilingual-e5)     │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │        Cached Embeddings             │
                    │   H_q: [B, 768]  H_d: [B, P, 768]    │
                    └──────────────────┬───────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         │                             │                             │
         ▼                             ▼                             ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  W + σ(a₁⊗b₁)  │         │  W + σ(a₂⊗b₂)  │   ...   │  W + σ(aₘ⊗bₘ)  │
│  Perturbation 1 │         │  Perturbation 2 │         │  Perturbation M │
└────────┬────────┘         └────────┬────────┘         └────────┬────────┘
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   NDCG@k = f₁   │         │   NDCG@k = f₂   │   ...   │   NDCG@k = fₘ   │
│  (Black-box!)   │         │  (Black-box!)   │         │  (Black-box!)   │
└────────┬────────┘         └────────┬────────┘         └────────┬────────┘
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────────────┐
                    │         Fitness Shaping              │
                    │   rank_transform([f₁, f₂, ..., fₘ])  │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │         Antithetic Pairing           │
                    │        Δfⱼ = (f⁺ⱼ - f⁻ⱼ) / 2         │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │           ES Update Rule             │
                    │  W ← W + α·(1/M)·Σ Δfⱼ·(aⱼ ⊗ bⱼ)    │
                    └──────────────────────────────────────┘
```

## Core Components

### 1. Rank-1 Noise Generator (`src/eggroll/noise.py`)

Instead of generating full perturbation matrices, we use **rank-1 decomposition**:

```
E = a ⊗ b^T    where a ∈ ℝ^{D_out}, b ∈ ℝ^{D_in}
```

This reduces memory from O(D_out × D_in) to O(D_out + D_in) per perturbation.

**Antithetic Sampling**: For each direction (a, b), we evaluate both +σE and -σE, which:
- Reduces variance by 2×
- Guarantees zero-mean perturbations
- Requires only M noise samples for 2M evaluations

### 2. Vectorized Scorer (`src/eggroll/scoring.py`)

Computes all 2M perturbed scores **without materializing the perturbation matrices**:

```python
# For W_i = W + σ·aᵢ⊗bᵢᵀ, the score becomes:
score_i = base + σ·(s_q·a_d + s_d·a_q) + σ²·s_q·s_d·‖a‖²

# Where:
s_q = H_q @ b    # Query projection onto noise direction
s_d = H_d @ b    # Doc projection onto noise direction
a_q = q_base @ a # Query embedding dot product with a
a_d = d_base @ a # Doc embedding dot product with a
```

This makes the forward pass O(B × P × M × D) instead of O(B × P × M × D²).

### 3. GPU-Accelerated NDCG (`src/eggroll/ndcg.py`)

Fully vectorized NDCG computation:
- Uses `topk` instead of full sort (O(P log k) vs O(P log P))
- Caches IDCG per query (constant across perturbations)
- Batches across all perturbations simultaneously

### 4. Fitness Shaping (`src/eggroll/shaping.py`)

Transforms raw NDCG values to stabilize gradient estimates:

| Method | Formula | Effect |
|--------|---------|--------|
| **Rank** | `(rank / (N-1)) - 0.5` | Uniform in [-0.5, 0.5], robust to outliers |
| **Z-score** | `(f - μ) / σ` | Zero-mean, unit variance |
| **Combined** | `rank(zscore(f))` | Best of both |

### 5. Adaptive Sigma (`src/eggroll/update.py`)

Automatically adjusts perturbation scale based on fitness variance:

```python
if variance < target * 0.5:
    σ *= (1 + rate)  # Increase exploration
elif variance > target * 2.0:
    σ *= (1 - rate)  # Decrease noise
```

## Experimental Results

We conducted experiments comparing EGGROLL (Evolution Strategies) against traditional gradient-based contrastive learning on two datasets: English (NanoMSMARCO) and Korean (Ko-StrategyQA).

### Results Summary

#### English: NanoMSMARCO

| Method | Train NDCG@10 | Val NDCG@10 | Best Val | Notes |
|--------|---------------|-------------|----------|-------|
| **EGGROLL (ES)** | 0.3222 | 0.1257 | **0.1540** | Healthy train-val gap |
| Baseline (Contrastive) | 1.0000 | 0.0972 | 0.1257 | Complete overfit |

#### Korean: Ko-StrategyQA

| Method | Train NDCG@10 | Val NDCG@10 | Best Val | Notes |
|--------|---------------|-------------|----------|-------|
| **EGGROLL (ES)** | 0.9183 | **0.7630** | **0.7697** | Strong generalization |
| Baseline (Contrastive) | 1.0000 | 0.7468 | 0.7527 | Overfit pattern |

---

### Key Findings

#### 1. EGGROLL Generalizes Better

The most striking observation is the **overfitting behavior** of gradient-based methods:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OVERFITTING COMPARISON                           │
└─────────────────────────────────────────────────────────────────────────┘

  Gradient-Based Contrastive:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Train NDCG: ████████████████████████████████████████████████ 1.0000  │
  │ Val NDCG:   █████████                                        0.0972  │
  └──────────────────────────────────────────────────────────────────────┘
  → Train=1.0 means PERFECT memorization of training data
  → Huge gap indicates severe overfitting

  EGGROLL (Evolution Strategies):
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Train NDCG: ████████████████                                 0.3222  │
  │ Val NDCG:   ██████                                           0.1257  │
  └──────────────────────────────────────────────────────────────────────┘
  → Modest train score indicates the model is NOT memorizing
  → Smaller train-val gap indicates better generalization
```

**Why does this happen?**

| Method | Optimization Target | Side Effect |
|--------|---------------------|-------------|
| **Contrastive** | Maximize similarity for positive pairs, minimize for negatives | Learns to perfectly separate train examples (memorization) |
| **EGGROLL** | Maximize NDCG directly via noisy exploration | Noise injection acts as implicit regularization |

The gradient-based baseline achieves **Train NDCG = 1.0** on both datasets — this is a red flag indicating the model has perfectly memorized the training data rather than learning generalizable patterns.

#### 2. EGGROLL Wins on Validation

Despite lower training scores, EGGROLL achieves **higher validation performance**:

| Dataset | EGGROLL Best Val | Baseline Best Val | Improvement |
|---------|------------------|-------------------|-------------|
| English (NanoMSMARCO) | 0.1540 | 0.1257 | **+22.5%** |
| Korean (Ko-StrategyQA) | 0.7697 | 0.7527 | **+2.3%** |

This confirms our hypothesis: **directly optimizing the target metric (NDCG) leads to better generalization than optimizing a surrogate loss (contrastive).**

#### 3. Cross-Lingual Consistency

The same pattern holds across both English and Korean:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CROSS-LINGUAL RESULTS                                │
└─────────────────────────────────────────────────────────────────────────┘

  English (NanoMSMARCO)                    Korean (Ko-StrategyQA)
  ┌────────────────────────┐               ┌────────────────────────┐
  │ EGGROLL    → 0.1540    │               │ EGGROLL    → 0.7697    │
  │ Baseline   → 0.1257    │               │ Baseline   → 0.7527    │
  │ Winner: EGGROLL +22.5% │               │ Winner: EGGROLL +2.3%  │
  └────────────────────────┘               └────────────────────────┘
```

This demonstrates that:
- Evolution strategies avoid the gradient-based overfitting trap **regardless of language**
- Rank-1 ES with fitness shaping works well as a **black-box optimizer** for non-differentiable NDCG
- The approach is **language-agnostic** — no language-specific tuning required

---

### Why Evolution Strategies Prevent Overfitting

| Factor | Gradient-Based | Evolution Strategies |
|--------|----------------|---------------------|
| **Update Signal** | Exact gradients → sharp minima | Noisy estimates → flat minima |
| **Exploration** | Follows loss surface precisely | Random perturbations explore broadly |
| **Implicit Regularization** | None (must add explicitly) | Built-in through noise injection |
| **Objective** | Surrogate loss (contrastive) | True metric (NDCG) |

The **noise injection** in ES acts as a form of **implicit regularization**, similar to dropout or weight noise, but applied at the optimization level rather than the architecture level.

---

### Hyperparameter Sensitivity

Best configurations found through grid search:

| σ (noise) | Population | LR | Head Dim | Best Val NDCG |
|-----------|------------|-----|----------|---------------|
| 0.05 | 256 | 0.2 | 128 | 0.1795 |
| 0.1 | 256 | 0.1 | 128 | 0.1538 |
| 0.02 | 256 | 0.1 | 128 | 0.1511 |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/eggroll-ndcg.git
cd eggroll-ndcg

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers
- Datasets
- Sentence-Transformers
- tqdm

## Usage

### Quick Start

```bash
# Run comparison experiment (English)
python scripts/compare_methods.py --num_steps 1000 --device mps

# Run comparison experiment (Korean)
python scripts/compare_korean.py --num_steps 1000 --device mps

# Full EGGROLL training
python -m src.train --task NanoMSMARCO --num_steps 5000 --device cuda
```

### Programmatic Usage

```python
from src.train import EGGROLLTrainer, TrainConfig

config = TrainConfig(
    task="NanoMSMARCO",
    num_steps=5000,
    population_size=256,
    sigma=0.05,
    learning_rate=0.1,
    head_output_size=128,
    device="cuda"
)

trainer = EGGROLLTrainer(config)
results = trainer.train()
print(f"Final Val NDCG: {results['val/ndcg@20']:.4f}")
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task` | `NanoMSMARCO` | NanoBEIR task name |
| `population_size` | 256 | Number of perturbations (must be even) |
| `sigma` | 0.02 | Initial noise scale |
| `learning_rate` | 0.05 | ES update step size |
| `head_output_size` | 256 | Projection head dimension |
| `ndcg_k` | 20 | NDCG cutoff |
| `adaptive_sigma` | True | Enable adaptive noise scaling |
| `shaping_method` | `rank` | Fitness shaping (`rank`, `zscore`, `combined`) |

## Project Structure

```
eggroll-ndcg/
├── src/
│   ├── eggroll/           # Core EGGROLL components
│   │   ├── noise.py       # Rank-1 noise generator
│   │   ├── scoring.py     # Vectorized score computation
│   │   ├── ndcg.py        # GPU-accelerated NDCG
│   │   ├── shaping.py     # Fitness shaping
│   │   └── update.py      # ES weight updates
│   ├── model/
│   │   ├── encoder.py     # ModernBERT encoder
│   │   ├── multilingual_encoder.py  # multilingual-e5
│   │   └── head.py        # Projection head
│   ├── data/
│   │   ├── load_nanobeir.py      # NanoBEIR loader
│   │   ├── load_ko_strategyqa.py # Korean dataset loader
│   │   ├── build_pools.py        # Candidate pool builder
│   │   └── cache_prehead.py      # Embedding cache
│   ├── baselines/
│   │   └── contrastive_trainer.py # Gradient baseline
│   ├── train.py           # Main trainer
│   └── eval.py            # Evaluation utilities
├── scripts/
│   ├── compare_methods.py # English comparison
│   ├── compare_korean.py  # Korean comparison
│   └── run_experiment.py  # Full experiment runner
├── configs/               # YAML configurations
├── tests/                 # Unit tests
└── cache/                 # Cached embeddings
```

## How It Works: Step by Step

### Training Loop

1. **Sample Noise**: Generate M rank-1 directions (a_j, b_j)
2. **Compute Scores**: Vectorized forward pass for all 2M perturbations
3. **Evaluate NDCG**: Black-box fitness evaluation
4. **Shape Fitness**: Rank transform to [-0.5, 0.5]
5. **Antithetic Diff**: Compute (f⁺ - f⁻) / 2 for variance reduction
6. **Update Weights**: W ← W + α · (1/M) · Σ Δf_j · (a_j ⊗ b_j^T)
7. **Adapt Sigma**: Adjust noise based on fitness variance

### Why It Works

1. **No Surrogate Loss**: We optimize NDCG directly, not a proxy
2. **Implicit Regularization**: Noise injection prevents sharp minima
3. **Rank-1 Efficiency**: Low-rank perturbations are memory-efficient
4. **Antithetic Variance Reduction**: Paired sampling cuts variance in half

## Comparison with Alternatives

| Method | Differentiable? | Optimizes NDCG? | Memory | Speed |
|--------|-----------------|-----------------|--------|-------|
| Contrastive | Yes | No (surrogate) | O(B×D²) | Fast |
| ListNet | Yes | Approximate | O(B×D²) | Fast |
| LambdaRank | Partially | Approximate | O(B×D²) | Medium |
| **EGGROLL** | No | **Yes (exact)** | O(M×D) | Medium |

## Future Works

### 1. Genetic Algorithm Extensions

EGGROLL's gradient-free nature opens doors to **true evolutionary algorithms**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Genetic EGGROLL Pipeline                      │
└─────────────────────────────────────────────────────────────────┘

  Generation t                              Generation t+1
  ┌─────────┐                               ┌─────────┐
  │ W₁ (fit=0.85) │──┐                      │ W'₁     │
  │ W₂ (fit=0.82) │──┼── Selection ──┐      │ W'₂     │
  │ W₃ (fit=0.79) │──┘               │      │ W'₃     │
  │ W₄ (fit=0.71) │ ✗                ├──→   │ W'₄     │
  │ W₅ (fit=0.65) │ ✗   Crossover ───┤      │ W'₅     │
  └─────────┘         + Mutation ────┘      └─────────┘
```

**Proposed approach:**
- **Tournament Selection**: Keep top-k performers
- **Crossover**: `W_child = α·W_parent1 + (1-α)·W_parent2`
- **Mutation**: `W' = W + σ·ε` (current EGGROLL perturbation)
- **Elitism**: Always preserve the best individual

This could provide better exploration of the weight space compared to pure ES.

### 2. Native FP4 Quantized Training

Since EGGROLL only requires **forward passes** (no backward pass, no gradient accumulation), we can potentially train directly in low-precision formats:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FP4 Native Training Loop                      │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  W (FP4)     │────→│ Forward Pass │────→│  NDCG Score  │
  │  4-bit       │     │  FP4 decode  │     │  (fitness)   │
  └──────────────┘     └──────────────┘     └──────────────┘
         ▲                                         │
         │                                         ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  Quantize    │←────│   Crossover  │←────│  Selection   │
  │  back to FP4 │     │   + Mutate   │     │  (top-k)     │
  └──────────────┘     └──────────────┘     └──────────────┘
```

**Why this matters:**
- **No gradient accumulation** = no need for FP16/FP32 master weights
- **4× memory reduction** compared to FP16 training
- **Native quantization-aware** = no post-training quantization loss
- Perfect for **edge device deployment** (train where you deploy)

**Implementation sketch:**
```python
# Pseudocode for FP4 native training
for generation in range(num_generations):
    # Decode FP4 weights to FP16 for forward pass
    W_fp16 = dequantize_nvfp4(W_fp4)
    
    # Evaluate fitness (NDCG) for population
    fitness = evaluate_population(W_fp16, perturbations)
    
    # Selection + Crossover in FP16
    W_new_fp16 = evolve(W_fp16, fitness)
    
    # Quantize back to FP4
    W_fp4 = quantize_nvfp4(W_new_fp16)
```

### 3. Soft Labeling for False Negatives

Current IR datasets suffer from **incomplete annotations** — many relevant documents are labeled as negatives simply because annotators didn't see them.

**Proposed: Ensemble-based soft labeling**

```python
# Use multiple embedding models to identify likely false negatives
models = [bge_large, e5_large, gte_large, instructor]

def compute_soft_labels(query, candidates, hard_labels):
    votes = []
    for model in models:
        scores = model.score(query, candidates)
        top_k = scores.topk(k=20).indices
        votes.append(top_k)
    
    # Documents retrieved by multiple models but labeled negative
    # are likely false negatives
    consensus = find_common_retrievals(votes)
    false_neg_candidates = consensus & (hard_labels == 0)
    
    soft_labels = hard_labels.clone()
    soft_labels[false_neg_candidates] = 0.5  # Partial credit
    
    return soft_labels
```

**Modified NDCG with soft labels:**
```
DCG = Σ (2^rel_i - 1) / log2(i + 2)

where rel_i ∈ {0, 0.5, 1, 2} instead of {0, 1, 2}
```

This gives partial credit for retrieving "hard negatives" that are actually relevant, reducing noise in the training signal.

### 4. Hybrid Training: ES Warmup + Gradient Fine-tuning

Even if EGGROLL alone doesn't match SFT performance, it could serve as a **better initialization**:

```
Phase 1: EGGROLL (gradient-free)     Phase 2: Gradient fine-tuning
┌─────────────────────────────┐     ┌─────────────────────────────┐
│  • Explore broadly          │     │  • Exploit local optima     │
│  • No gradient noise        │────→│  • Fast convergence         │
│  • Find good basin          │     │  • Polish final performance │
│  • 1000-2000 steps          │     │  • 100-500 steps            │
└─────────────────────────────┘     └─────────────────────────────┘
```

**Hypothesis**: EGGROLL finds flatter minima (due to noise injection), which could lead to better generalization even after gradient fine-tuning.

### 5. Multi-Objective Evolution

Extend EGGROLL to optimize multiple metrics simultaneously:

```python
fitness = (
    α * ndcg_score +           # Relevance
    β * diversity_score +       # Result diversity  
    γ * freshness_score +       # Temporal relevance
    δ * fairness_score          # Exposure fairness
)
```

Evolution strategies naturally handle multi-objective optimization through Pareto frontiers.

### Research Questions

1. **Can genetic crossover improve over pure ES?** Crossover provides a fundamentally different exploration mechanism.

2. **Is FP4 precision sufficient for ES training?** Noise injection might mask quantization errors.

3. **Does soft labeling improve generalization?** Reducing false negative penalty could help on unseen queries.

4. **What's the optimal ES→Gradient handoff point?** When does gradient fine-tuning start to overfit?

## Citation

If you use this code, please cite:

```bibtex
@software{eggroll_ndcg,
  title = {EGGROLL-NDCG: Evolution-Guided Gradient-free Ranking Optimization},
  year = {2025},
  url = {https://github.com/yourusername/eggroll-ndcg}
}
```

## License

MIT License

## Acknowledgments

- NanoBEIR datasets from MTEB
- Ko-StrategyQA from MTEB Korean benchmarks
- ModernBERT from Answer.AI
- multilingual-e5-base from Microsoft
