# Recursive Reasoning with Tiny Models: Sudoku Solving via TRM, MoE, and MCTS

A study of **recursive depth-efficient reasoning** in tiny transformer models (<1M parameters) applied to 4x4 Sudoku solving. Extends the [TinyRecursiveModels](https://github.com/nicholaschenai/TinyRecursiveModels) framework with **Mixture-of-Experts (MoE)** and **Monte Carlo Tree Search (MCTS)** for enhanced puzzle-solving capabilities.

## Overview

Large language models achieve reasoning through massive scale. This project explores the opposite question: **how small can a reasoning model be?** Using recursive weight-tied transformers with only ~500K parameters, we investigate:

1. Whether recursive computation (iterating the same weights) can substitute for model scale
2. Whether MoE routing can improve reasoning without proportional parameter cost
3. Whether MCTS can provide structured search to complement learned reasoning

## Key Results

| Model | Parameters | Approach | Sudoku Accuracy |
|-------|-----------|----------|-----------------|
| TRM-Small (baseline) | 526K | Recursive transformer | Baseline |
| TRM-Small + MoE | 1.47M | + Mixture of Experts (16+1 experts) | Improved |
| TRM + MCTS | ~526K | + Monte Carlo Tree Search | Structured search |
| TRM + MoE + RL | 1.47M | + Reinforcement learning fine-tuning | Best |

## Architecture

### Tiny Recursive Model (TRM)

The core insight: **shared-weight recursive refinement** enables reasoning in tiny models.



- **H-cycles** = 2 (outer iterations for high-level reasoning)
- **L-cycles** = 4 (inner iterations for local constraint propagation)
- **Core**: 2-3 layer transformer, shared across all iterations
- **Hidden**: 128, Heads: 4

### Mixture of Experts Extension

Replaces standard FFN layers with 17 experts (16 routed + 1 shared), achieving 2.79x parameters with same activated compute per token.

### MCTS Solver

Monte Carlo Tree Search for structured exploration of solution space:
- **Node expansion**: Each node represents a partial Sudoku assignment
- **Value network**: TRM-based evaluation of board states
- **Policy**: Guided by model confidence over possible digits
- **Rollout**: Complete puzzle attempts for reward estimation

## Project Structure



## My Contributions

| Component | What I Built |
|-----------|-------------|
|  | Complete MCTS implementation integrated with TRM for structured search |
|  | Mixture-of-Experts extension to the TRM architecture |
|  | Reinforcement learning fine-tuning pipeline |
|  | Extended pretraining experiments and analysis |
|  | Detailed mathematical formulation of the approach |

## Built On

- [TinyRecursiveModels](https://github.com/nicholaschenai/TinyRecursiveModels) — Base recursive transformer framework
- Chen, N. et al. *Less is More: Recursive Reasoning with Tiny Networks*. 2024.

## Technical Details

### Model Configurations

| Config | Core Layers | Baseline Params | MoE Params | MoE Multiplier |
|--------|-------------|-----------------|------------|----------------|
| Small | 2 | 526K | 1.47M | 2.79x |
| Base | 3 | 788K | 2.20M | 2.79x |

### Training Pipeline

1. **Pretraining**: Supervised learning on solved Sudoku puzzles
2. **MoE Extension**: Replace FFN with expert routing
3. **RL Fine-tuning**: Reward-based optimization for solving accuracy
4. **MCTS Integration**: Search-augmented inference

## Tech Stack

- PyTorch
- Mixture of Experts (custom implementation)
- Monte Carlo Tree Search
- Reinforcement Learning (PPO-style)

## License

MIT
