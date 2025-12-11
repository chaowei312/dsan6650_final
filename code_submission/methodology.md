# Methodology

## 1. Tiny Recursive Reasoning Model (TRM) Architecture

### 1.1 TRM Update Rule

The TRM uses a hierarchical recursive structure with two levels of computation: H-cycles (outer) and L-cycles (inner).

**Input:** A 4×4 Sudoku puzzle where clues are digits 1-4 and empty cells are marked. Each cell is tokenized and embedded:
$$
x = \text{token\_emb}(\text{puzzle}) + \text{pos\_emb} \quad \in \mathbb{R}^{16 \times d}
$$

**Latent States:** $z_H, z_L \in \mathbb{R}^{16 \times d}$ are initialized from learned parameters (`H_init`, `L_init`), then iteratively refined:

$$
\text{For each } h = 1, \ldots, H:
$$
$$
\quad \text{For each } l = 1, \ldots, L: \quad z_L \leftarrow f_L(z_L + z_H + x)
$$
$$
\quad z_H \leftarrow f_H(z_H + z_L)
$$

**Output:** The final $z_H$ encodes the solution. A linear head decodes it to token logits:
$$
\hat{y} = \text{LM\_head}(z_H) \quad \in \mathbb{R}^{16 \times 6}
$$

Where:
- $H$ = number of H-cycles (outer iterations) = **2**
- $L$ = number of L-cycles (inner iterations) = **4**
- $f_L, f_H$ = shared Core reasoning module (transformer stack)
- $d$ = hidden size = **128**, Attention heads = **4**

> **Key insight:** $f_L$ and $f_H$ share the **same weights** (`L_level` in code), enabling recursive refinement through weight tying. The puzzle embedding $x$ is injected at every L-cycle, anchoring the reasoning to the original constraints.

### 1.3 Model Configurations & Parameter Counts

| Config | Core Layers | Baseline Params | MoE Params | MoE Multiplier |
|--------|-------------|-----------------|------------|----------------|
| **Small** | L_layers=2 | **526K** | **1.47M** | 2.79× |
| **Base** | L_layers=3 | **788K** | **2.20M** | 2.79× |

**Architecture Settings (both configs):**
- H_cycles = 2, L_cycles = 4
- Hidden = 128, Heads = 4, Expansion = 4
- Vocab = 6 (pad, empty, 1-4)

**Parameter Breakdown (Small, L=2):**
| Component | Params | Description |
|-----------|--------|-------------|
| Embeddings | 768 | Token embeddings (6 × 128) |
| Core (L_level) | 524K | 2-layer transformer (shared for L/H cycles) |
| LM Head | 768 | Output projection (128 → 6) |
| Q Head | 258 | ACT halting head (128 → 2) |

**MoE Overhead:** The 2.79× parameter increase comes from replacing each FFN with 17 smaller experts (16 routed + 1 shared), while maintaining the same activated compute per token.

### 1.2 Adaptive Computation Time (ACT)

The ACT mechanism allows the model to dynamically decide when to halt computation. A Q-head predicts a halting score:

$$
q_{\text{halt}} = \text{Q\_head}(z_H[:, 0])
$$

**Halting Condition:**
$$
\text{halt} = \mathbb{1}[q_{\text{halt}} > 0]
$$

**Interpretation:** $\sigma(q_{\text{halt}}) \approx P(\text{solution is good})$

#### ACT Loss - Unified Formula

Both pretraining and RL fine-tuning use the same supervised BCE loss structure:

$$
\mathcal{L}_{\text{ACT}} = \text{BCE}\left(\sigma(q_{\text{halt}}), \text{target}\right)
$$

| Phase | Target | Meaning | Sparsity |
|-------|--------|---------|----------|
| **Pretraining** | $\mathbb{1}[\text{all argmax tokens match GT}]$ | Must match ONE specific solution | Very sparse |
| **RL Fine-tuning** | $\mathbb{1}[R = 1]$ | ANY valid solution (0 violations) | Denser |

**Pretraining target** (exact token match to ONE ground truth):
```python
preds = logits.argmax(dim=-1)  # Greedy decode
seq_is_correct = ((preds == solutions) | ~mask).all(dim=1).float()  # 1 if ALL match GT
```

**RL target** (rule-based, accepts ANY valid solution):
```python
halt_target = (rewards > 0.99).float()  # 1 if R=1 (any valid solution)
```

**Key difference:** Pretraining requires matching the ONE ground truth, while RL accepts ANY of the potentially multiple valid solutions. This makes RL's signal denser.

#### ACT Coefficients

| Phase | Coefficient |
|-------|-------------|
| Pretraining | 0.5 |
| RL Fine-tuning | 0.1 |

*(Source: `pretraining.ipynb`, `rlfinetune/reinforce.py`)*

---

## 2. RL Fine-tuning Methods

All RL methods share a common objective structure with method-specific components.

### 2.1 REINFORCE + KL Divergence

**Loss Function:**
$$
\mathcal{L}_{\text{REINFORCE+KL}} = -\mathbb{E}\left[\log \pi_\theta(a|s) \cdot (R - b)\right] + \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}}) - \lambda_H H(\pi_\theta) + \lambda_{\text{ACT}} \mathcal{L}_{\text{ACT}}
$$

Where:
- $R$ = reward from Sudoku constraint violations
- $b$ = baseline (mean reward over batch)
- $\beta = 0.1$ (KL coefficient)
- $H(\pi_\theta)$ = **entropy** of policy distribution (encourages exploration)
- $\lambda_H = 0.01$ (entropy coefficient)
- $\lambda_{\text{ACT}} = 0.1$ (ACT loss coefficient)
- $\pi_{\text{ref}}$ = frozen reference policy (copy of initial model)

**Entropy term:** $H(\pi_\theta) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$ — higher entropy = more exploration

**KL Divergence:**
$$
D_{KL}(\pi_\theta \| \pi_{\text{ref}}) = \sum_{a} \pi_\theta(a|s) \log \frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)}
$$

### 2.2 PPO + KL Divergence

**Clipped Surrogate Objective:**
$$
\mathcal{L}_{\text{PPO}} = -\mathbb{E}\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]
$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ (probability ratio)
- $\epsilon = 0.2$ (clipping parameter)
- $A_t = R_t - b$ (normalized advantages)

**Full PPO+KL Loss:**
$$
\mathcal{L}_{\text{PPO+KL}} = \mathcal{L}_{\text{PPO}} + \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}}) - \lambda_H H(\pi_\theta) + \lambda_{\text{ACT}} \mathcal{L}_{\text{ACT}}
$$

**PPO Epochs:** 4 optimization passes per batch

### 2.3 GRPO + KL Divergence (Group Relative Policy Optimization)

**Key Idea:** Use relative comparison within a group instead of absolute rewards.

For each puzzle, sample $K=16$ solutions. Two normalization approaches:

**Standard GRPO (reward-based):**
$$
A_k = \frac{R_k - \mu}{\sigma} \quad \text{where } \mu = \text{mean}(R), \sigma = \text{std}(R)
$$

**Our Implementation (rank-based variant):**
$$
\text{rank}_k \in \{0, 1, \ldots, K-1\}, \quad A_k = \frac{2 \cdot \text{rank}_k}{K-1} - 1 \in [-1, +1]
$$

Rank-based is more robust to reward scale/outliers but loses magnitude information.

**GRPO Loss:**
$$
\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\log \pi_\theta(a_k|s) \cdot A_k\right]
$$

**Full GRPO+KL Loss:**
$$
\mathcal{L}_{\text{GRPO+KL}} = \mathcal{L}_{\text{GRPO}} + \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}}) - \lambda_H H(\pi_\theta) + \lambda_{\text{ACT}} \mathcal{L}_{\text{ACT}}
$$

---

## 3. Reward Structure for RL Fine-tuning

The reward is based on **Sudoku constraint violations** (no ground truth needed):

$$
R = \left(1 - \frac{V}{24}\right) \cdot \mathbb{1}[\text{clues preserved}]
$$

**Violation counting** (for 4×4 Sudoku):
- **Row violations:** For each row and each digit (1-4), +1 if digit appears more than once
- **Column violations:** For each column and each digit, +1 if digit appears more than once  
- **Box violations:** For each 2×2 box and each digit, +1 if digit appears more than once

**Maximum violations = 24** (not 48), because:
- Each row/col/box has only 4 cells, so at most **2 different digits** can appear >1 time
- Max per row: 2 violations × 4 rows = 8
- Max per col: 2 violations × 4 cols = 8
- Max per box: 2 violations × 4 boxes = 8
- **Total max: 8 + 8 + 8 = 24**

**Reward values:**

| Violations (V) | Reward (R) | Meaning |
|----------------|------------|---------|
| 0 | **+1.0** | Valid solution |
| 6 | +0.75 | Few violations |
| 12 | +0.5 | Moderate violations |
| 24 | **0.0** | Maximum violations |

**Reward range:** $R \in [0, 1]$ when clues preserved (since max V = 24 gives R = 0).

**Clue preservation:** If any given clue is overwritten, $R = 0$ regardless of violations.

*(Implemented in `rlfinetune/rewards.py::compute_sudoku_reward`)*

---

## 4. Validity Testing Schema

### 4.1 Definition of Valid Prediction

A predicted cell value $v$ at position $(r, c)$ is **valid** if and only if:

1. **Value Range:** $v \in \{1, 2, 3, 4\}$ for 4×4 Sudoku
2. **Row Uniqueness:** $v \notin \{board[r, c'] : c' \neq c\}$
3. **Column Uniqueness:** $v \notin \{board[r', c] : r' \neq r\}$
4. **Box Uniqueness:** $v \notin \{board[r', c'] : (r', c') \in \text{same 2×2 box}, (r', c') \neq (r, c)\}$

### 4.2 Evaluation Metrics

**Prediction Validity Rate:**
$$
\text{Validity} = \frac{\text{Valid Predictions}}{\text{Total Predictions}} \times 100\%
$$

**Solve Rate:**
$$
\text{Solve Rate} = \frac{\text{Puzzles with ALL valid predictions}}{\text{Total Puzzles}} \times 100\%
$$

**Standard Error:**
$$
SE = \sqrt{\frac{p(1-p)}{n}} \times 100\%
$$

*(Implemented in `tests/evaluation.py::count_prediction_validity`)*

---

## 5. Experiment Settings

### 5.1 Dataset Generation

**4×4 Sudoku Puzzles** generated on-the-fly:
1. Generate a complete valid 4×4 Sudoku solution
2. Randomly select $k$ cells to mask as empty
3. Empty cells encoded as token `1`, digits 1-4 encoded as tokens `2-5`

**Training Difficulties:** $k \in \{4, 6, 8, 10, 12\}$ empty cells

**Testing Difficulties (Interpolation):** $k \in \{5, 7, 9, 11\}$ empty cells

> **Key Insight:** Models are trained on even-numbered difficulties but tested on odd-numbered difficulties they never saw during training.

### 5.2 Pretraining Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch Size | 32 |
| Batches per Epoch | 100 |
| **Total Samples** | **320,000** |
| Learning Rate | 1e-4 |
| Weight Decay | 0.01 |
| Optimizer | AdamW |

### 5.3 RL Fine-tuning Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch Size | 32 |
| Batches per Epoch | 30 |
| **Total Samples** | **48,000** |
| Learning Rate | 1e-4 |
| Weight Decay | 0.01 |
| KL Coefficient (β) | 0.1 |
| Entropy Coefficient | 0.01 |
| ACT Coefficient | 0.1 |
| PPO Epochs | 3 |
| GRPO Samples (K) | 16 |

### 5.4 Sample Size Comparison

| Phase | Samples | Purpose |
|-------|---------|---------|
| Pretraining | 320,000 | Learn basic Sudoku solving |
| RL Fine-tuning | 48,000 | Refine with rule-based rewards |
| **Ratio** | **6.67:1** | Pretraining >> RL |

---

## 6. Curriculum Training Setup

### 6.1 Gaussian Curriculum with Minimum Floor

The curriculum uses a **Gaussian-weighted difficulty distribution** that shifts from easy to hard:

$$
w_i^{\text{raw}} = \exp\left(-\frac{(i - \mu(t))^2}{2\sigma^2}\right)
$$

Where:
- $i \in \{0, 1, 2, 3, 4\}$ indexes difficulties $\{4, 6, 8, 10, 12\}$
- $\mu(t) = 1.5 + 2.5 \cdot \min(1, t/0.8T)$ (center shifts from 1.5 to 4.0)
- $\sigma = 1.2$ (Gaussian width)
- $T$ = total epochs

**Minimum Floor:** Each difficulty maintains at least 5% weight:
$$
w_i = 0.05 + 0.75 \cdot \frac{w_i^{\text{raw}}}{\sum_j w_j^{\text{raw}}}
$$

### 6.2 Curriculum Progression

| Epoch | Center | Focus |
|-------|--------|-------|
| 1 | 1.5 | Easy-Medium (4-6 empty) |
| 40 | 3.0 | Medium (8 empty) |
| 80+ | 4.0 | Hard (10-12 empty) |

*(Implemented in `pretraining.ipynb::get_curriculum_weights` and `rl_finetune.ipynb::get_difficulty_weights`)*

---

## 7. Mixture of Experts (MoE) Configuration

### 7.1 DeepSeek-Style MoE Architecture

Our MoE implementation follows the **DeepSeek MoE** design with a shared/persistent expert:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Routed Experts** | 16 | Pool of specialized experts |
| **Top-K** | 4 | Experts selected per token |
| **Shared Expert** | 1 | Always-on persistent expert |
| **Total Activated** | 5 | 1 shared + 4 routed |

### 7.2 Expert Sizing (Compute-Neutral Design)

Each expert has reduced size to maintain **constant compute**:

$$
\text{Expert Size} = \frac{\text{Original FFN}}{1 + \text{top\_k}} = \frac{\text{FFN}}{5}
$$

**Compute Analysis:**
- Activated compute = $(1 + 4) \times \frac{\text{FFN}}{5} = 1\times$ original ✓
- Total params = $(16 + 1) \times \frac{\text{FFN}}{5} = 3.4\times$ FFN params

This gives **3.4× more capacity** with **same inference cost**.

### 7.3 Load Balancing Loss

To prevent expert collapse (all tokens routed to few experts), we use auxiliary load balancing loss:

$$
\mathcal{L}_{\text{aux}} = N \cdot \sum_{i=1}^{N} p_i \cdot f_i
$$

Where:
- $N$ = number of routed experts (16)
- $p_i$ = average routing probability for expert $i$: $\frac{1}{T}\sum_t P(\text{expert } i | x_t)$
- $f_i$ = fraction of tokens assigned to expert $i$: $\frac{|\{t : i \in \text{top-k}(x_t)\}|}{T \cdot K}$

**Coefficient:** $\lambda_{\text{aux}} = 0.01$

**Effect:** Encourages uniform expert utilization. If experts are perfectly balanced, $p_i = f_i = \frac{1}{N}$, giving $\mathcal{L}_{\text{aux}} = 1$.

### 7.4 Router with Jitter Noise

During training, input is jittered to encourage exploration:

$$
x' = x \cdot \text{Uniform}(1 - \epsilon, 1 + \epsilon), \quad \epsilon = 0.01
$$

*(Implemented in `models/moe.py::MoELayerOptimized`)*

---

## 8. Experiment Results

### 8.1 Pretraining Results (Interpolation Test)

Testing on **unseen difficulties** [5, 7, 9, 11] with n=300 puzzles per difficulty:

| Model | Avg Validity | 5 empty | 7 empty | 9 empty | 11 empty | Avg ACT Steps |
|-------|-------------|---------|---------|---------|----------|---------------|
| **Baseline** | 90.4% | 99.5% ±0.2 | 97.9% ±0.3 | 89.9% ±0.6 | 74.3% ±0.8 | 2.1 |
| **Curriculum** | 92.2% | 99.8% ±0.1 | 98.5% ±0.3 | 92.8% ±0.5 | 77.6% ±0.7 | 3.3 |
| **MoE** | 89.3% | 99.1% ±0.2 | 98.0% ±0.3 | 88.5% ±0.6 | 71.6% ±0.8 | 3.3 |

**Fully Solved Rate:**

| Model | Avg Solve | 5 empty | 7 empty | 9 empty | 11 empty |
|-------|----------|---------|---------|---------|----------|
| **Baseline** | 77.9% | 99.0% | 95.7% | 74.3% | 42.7% |
| **Curriculum** | 83.1% | 99.7% | 97.0% | 84.0% | 51.7% |
| **MoE** | 77.6% | 98.7% | 96.3% | 74.0% | 41.3% |

**Model Parameters (L_layers=2):**
- Baseline/Curriculum: 526,082
- MoE: 1,468,674 (2.79× larger)

*(Source: `tests/interpolation_test_results.json`, `tests/results.md`)*

### 8.2 RL Fine-tuning Results (Interpolation Test)

Testing on **unseen difficulties** [5, 7, 9, 11] with n=200 puzzles per difficulty:

| Method | Avg Validity | 5 empty | 7 empty | 9 empty | 11 empty | Δ Baseline |
|--------|-------------|---------|---------|---------|----------|------------|
| **Baseline** | 91.8% ±0.5 | 99.6% | 97.7% | 90.5% | 79.5% | — |
| **REINFORCE** | 89.7% ±0.7 | 97.8% | 94.1% | 89.3% | 77.7% | -2.1% ⚠️ |
| **REINFORCE+KL** | 94.3% ±0.5 | 99.1% | 98.3% | 94.5% | 85.1% | +2.4% ✅ |
| **PPO+KL** | 83.0% ±0.8 | 95.5% | 92.3% | 81.8% | 62.5% | -8.8% ⚠️ |
| **GRPO+KL** | **97.8%** ±0.2 | 100.0% | 100.0% | 97.7% | 93.6% | **+6.0%** ✅ |

**Solve Rate:**

| Method | Avg Solve | Δ Baseline |
|--------|----------|------------|
| **Baseline** | 82.2% | — |
| **REINFORCE** | 75.9% | -6.4% |
| **REINFORCE+KL** | 84.5% | +2.2% |
| **PPO+KL** | 66.8% | -15.5% |
| **GRPO+KL** | **95.2%** | **+13.0%** |

**ACT Steps (lower = more confident):**

| Method | Avg ACT | 5 empty | 7 empty | 9 empty | 11 empty |
|--------|---------|---------|---------|---------|----------|
| Baseline | 3.4 | 1.0 | 1.4 | 3.2 | 7.8 |
| REINFORCE | 1.0 | 1.0 | 1.0 | 1.0 | 1.1 |
| REINFORCE+KL | 1.7 | 1.0 | 1.1 | 1.8 | 2.9 |
| PPO+KL | 3.8 | 3.7 | 2.5 | 3.7 | 5.1 |
| GRPO+KL | 1.1 | 1.0 | 1.0 | 1.1 | 1.1 |

*(Source: `tests/results.md`)*

### 8.3 Reasoning Capability Analysis

To fairly compare reasoning ability across methods with different per-cell validity rates, we introduce **Reasoning Efficiency** — a metric that accounts for the ceiling effect when validity is high.

#### 8.3.1 The Problem with Raw Ratios

A naive approach compares actual solve rate to the "independent baseline" $v^n$ (validity rate raised to number of empty cells):

$$
\text{Reasoning Ratio} = \frac{\text{Solve Rate}}{v^n}
$$

However, this metric is **misleading** because:
1. Solve rate is capped at 100%
2. When validity is high (e.g., GRPO+KL at 100%), $v^n$ is also high, leaving little room for the ratio to grow
3. A model with perfect validity and perfect solve has ratio = 1.0, which incorrectly suggests "no reasoning"

#### 8.3.2 Reasoning Efficiency Metric

We define **Reasoning Efficiency** as the fraction of *achievable improvement* that the model captures:

$$
\text{Reasoning Efficiency} = \frac{\text{Solve Rate} - v^n}{1 - v^n}
$$

Where:
- $v^n$ = independent baseline (per-cell validity raised to number of empty cells)
- $1 - v^n$ = "room to improve" (gap from independent baseline to perfect)
- Numerator = actual improvement over independent baseline

**Interpretation:**
- **0%** = No reasoning beyond independent cell predictions
- **100%** = Perfect reasoning (captures all achievable improvement)
- **Negative** = Correlated errors (worse than independent predictions)

#### 8.3.3 Computed Reasoning Efficiency

**At 11 empty cells (hardest difficulty):**

| Method | Validity (v) | v¹¹ (baseline) | Solve Rate | Room (1-v¹¹) | Improvement | **Reasoning Efficiency** |
|--------|-------------|----------------|------------|--------------|-------------|--------------------------|
| **GRPO+KL** | 93.6% | 48.19% | 85.5% | 51.81% | +37.31 pts | **72.0%** 🏆 |
| Baseline | 79.5% | 8.93% | 54.5% | 91.07% | +45.57 pts | 50.0% |
| REINFORCE+KL | 85.1% | 17.40% | 58.0% | 82.60% | +40.60 pts | 49.2% |
| REINFORCE | 77.7% | 6.39% | 46.5% | 93.61% | +40.11 pts | 42.8% |
| PPO+KL | 62.5% | 0.35% | 30.5% | 99.65% | +30.15 pts | **30.3%** ⚠️ |

**At 9 empty cells:**

| Method | v⁹ (baseline) | Solve Rate | Room | **Reasoning Efficiency** |
|--------|---------------|------------|------|--------------------------|
| **GRPO+KL** | 81.01% | 95.5% | 18.99% | **76.3%** 🏆 |
| Baseline | 41.02% | 79.5% | 58.98% | 65.2% |
| REINFORCE+KL | 59.87% | 85.0% | 40.13% | 62.6% |
| REINFORCE | 36.43% | 74.5% | 63.57% | 59.9% |
| PPO+KL | 14.92% | 64.0% | 85.08% | 57.7% |

**At 7 empty cells:**

| Method | v⁷ (baseline) | Solve Rate | Room | **Reasoning Efficiency** |
|--------|---------------|------------|------|--------------------------|
| **GRPO+KL** | 100.0% | 100.0% | 0% | **100%** (perfect) 🏆 |
| Baseline | 85.0% | 95.5% | 15.0% | 70.0% |
| REINFORCE+KL | 88.69% | 96.5% | 11.31% | 69.1% |
| REINFORCE | 65.07% | 87.0% | 34.93% | 62.8% |
| PPO+KL | 56.79% | 83.5% | 43.21% | 61.8% |

**At 5 empty cells:**

| Method | v⁵ (baseline) | Solve Rate | Room | **Reasoning Efficiency** |
|--------|---------------|------------|------|--------------------------|
| **GRPO+KL** | 100.0% | 100.0% | 0% | **100%** (perfect) 🏆 |
| Baseline | 98.01% | 99.5% | 1.99% | 74.9% |
| REINFORCE+KL | 95.59% | 98.5% | 4.41% | 66.0% |
| REINFORCE | 89.43% | 95.5% | 10.57% | 57.4% |
| PPO+KL | 79.39% | 89.0% | 20.61% | 46.6% |

#### 8.3.4 Average Reasoning Efficiency Summary

| Method | Avg Reasoning Efficiency | Interpretation |
|--------|--------------------------|----------------|
| **GRPO+KL** | **~87%** | Best reasoner — captures most of available potential |
| Baseline | ~65% | Solid reasoning through more iterations (ACT: 7.8) |
| REINFORCE+KL | ~62% | Good reasoning, moderate efficiency |
| REINFORCE | ~56% | Moderate reasoning capability |
| **PPO+KL** | **~49%** | Weakest reasoning — fails to coordinate predictions |

#### 8.3.5 Key Insights

1. **GRPO+KL has genuinely superior reasoning**: At 11-empty, GRPO+KL captures 72% of possible improvement vs PPO+KL's 30%, even though PPO+KL had more "room" to improve. This confirms GRPO+KL's superior cross-cell coordination.

2. **High reasoning ratios can be misleading**: PPO+KL's "87x ratio" (solve/v¹¹) appears impressive but reflects a catastrophically low baseline (0.35%), not strong reasoning. The Reasoning Efficiency of 30.3% reveals the true weakness.

3. **Optimal strategy = High Validity × High Reasoning Efficiency**: GRPO+KL achieves the best absolute performance by combining near-perfect per-cell accuracy (93.6%) with strong reasoning coordination (72% efficiency).

4. **Baseline uses more compute to compensate**: The Baseline model achieves 50% reasoning efficiency at 11-empty, but requires 7.8 ACT steps vs GRPO+KL's 1.1 steps — trading compute for reasoning.

**Best Method:** GRPO+KL achieves the highest prediction validity (97.8%), solve rate (95.2%), and reasoning efficiency (~87%) on interpolation difficulties.

