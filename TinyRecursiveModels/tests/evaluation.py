"""
Evaluation utilities for Sudoku models.

This module provides standardized evaluation metrics for comparing
different training approaches (Baseline, Curriculum, MoE, RL-finetuned).

Metrics:
- Prediction Validity: % of model's predicted cells that satisfy Sudoku constraints
- Solve Rate: % of puzzles where ALL predictions are valid

Usage:
    from tests.evaluation import evaluate_model, compare_models, plot_comparison
    
    results = evaluate_model(model, device, num_samples=300)
    comparison = compare_models({'Baseline': model1, 'Curriculum': model2}, device)
    plot_comparison(comparison)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Results from evaluating a model at a single difficulty level."""
    prediction_validity: float  # % of predictions that are valid
    prediction_se: float        # Standard error of prediction validity
    solve_rate: float           # % of puzzles fully solved
    solve_se: float             # Standard error of solve rate
    num_samples: int            # Number of puzzles tested
    num_predictions: int        # Total predictions made
    difficulty: int             # Number of empty cells


def generate_puzzle(grid_size: int, num_empty: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a random Sudoku puzzle and its solution.
    
    Args:
        grid_size: Size of the grid (4 for 4x4 Sudoku)
        num_empty: Number of cells to leave empty
        
    Returns:
        (puzzle, solution) where puzzle has 1s for empty cells
    """
    def generate_solved(size):
        board = [[0] * size for _ in range(size)]
        
        def is_valid(board, row, col, num):
            if num in board[row]:
                return False
            if num in [board[r][col] for r in range(size)]:
                return False
            box_size = int(size ** 0.5)
            box_r, box_c = (row // box_size) * box_size, (col // box_size) * box_size
            for r in range(box_r, box_r + box_size):
                for c in range(box_c, box_c + box_size):
                    if board[r][c] == num:
                        return False
            return True
        
        def solve(board):
            for r in range(size):
                for c in range(size):
                    if board[r][c] == 0:
                        nums = list(range(1, size + 1))
                        np.random.shuffle(nums)
                        for num in nums:
                            if is_valid(board, r, c, num):
                                board[r][c] = num
                                if solve(board):
                                    return True
                                board[r][c] = 0
                        return False
            return True
        
        solve(board)
        return board
    
    solved = generate_solved(grid_size)
    solution = torch.tensor([[cell + 1 for cell in row] for row in solved]).flatten()
    
    puzzle = solution.clone()
    indices = torch.randperm(grid_size * grid_size)[:num_empty]
    puzzle[indices] = 1  # 1 = empty marker
    
    return puzzle, solution


def count_prediction_validity(board: torch.Tensor, mask: torch.Tensor, size: int = 4) -> Tuple[int, int]:
    """Count how many of the model's predictions are valid.
    
    A prediction is valid if it doesn't violate any Sudoku constraint
    (row, column, or box uniqueness).
    
    Args:
        board: Completed board with predictions (in digit space 1-4)
        mask: Boolean mask where True = model predicted this cell
        size: Grid size (4 for 4x4)
        
    Returns:
        (valid_predictions, total_predictions)
    """
    board = board.cpu().numpy() if torch.is_tensor(board) else board
    mask = mask.cpu().numpy() if torch.is_tensor(mask) else mask
    board = board.reshape(size, size)
    mask = mask.reshape(size, size)
    
    valid = 0
    total = int(mask.sum())
    box_size = int(size ** 0.5)
    
    for r in range(size):
        for c in range(size):
            if not mask[r, c]:
                continue
            
            val = board[r, c]
            is_valid = True
            
            # Check value range
            if not (1 <= val <= size):
                is_valid = False
            else:
                # Check row (excluding self)
                row_vals = [board[r, cc] for cc in range(size) if cc != c]
                if val in row_vals:
                    is_valid = False
                
                # Check column (excluding self)
                col_vals = [board[rr, c] for rr in range(size) if rr != r]
                if val in col_vals:
                    is_valid = False
                
                # Check box (excluding self)
                box_r, box_c = (r // box_size) * box_size, (c // box_size) * box_size
                box_vals = [board[rr, cc] 
                           for rr in range(box_r, box_r + box_size) 
                           for cc in range(box_c, box_c + box_size)
                           if (rr, cc) != (r, c)]
                if val in box_vals:
                    is_valid = False
            
            if is_valid:
                valid += 1
    
    return valid, total


def evaluate_model(
    model: torch.nn.Module,
    device: torch.device,
    num_empty: int,
    num_samples: int = 200,
    num_iterations: int = 8,
    grid_size: int = 4
) -> EvaluationResult:
    """Evaluate a model at a specific difficulty level.
    
    Args:
        model: The model to evaluate (must have initial_carry and forward methods)
        device: Device to run evaluation on
        num_empty: Number of empty cells (difficulty)
        num_samples: Number of puzzles to test
        num_iterations: Number of model iterations per puzzle
        grid_size: Size of Sudoku grid
        
    Returns:
        EvaluationResult with all metrics
    """
    model.eval()
    total_valid = 0
    total_predictions = 0
    fully_solved = 0
    
    for _ in range(num_samples):
        puzzle, solution = generate_puzzle(grid_size, num_empty)
        puzzle_t = puzzle.unsqueeze(0).to(device)
        batch = {
            'inputs': puzzle_t, 
            'puzzle_identifiers': torch.zeros(1, dtype=torch.long, device=device)
        }
        
        carry = model.initial_carry(batch)
        carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
        carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
        carry.halted = carry.halted.to(device)
        carry.steps = carry.steps.to(device)
        
        with torch.no_grad():
            for _ in range(num_iterations):
                carry, outputs = model(carry, batch)
        
        # Get predictions (token indices 2-5 map to digits 1-4)
        preds = outputs['logits'].argmax(dim=-1)[0]
        mask = (puzzle == 1)  # Empty cells
        
        # Complete puzzle with predictions
        completed = puzzle.clone()
        completed[mask] = preds[mask.to(device)].cpu()
        
        # Convert from token space (2-5) to digit space (1-4)
        completed_digits = completed - 1
        
        # Count valid predictions
        valid, total = count_prediction_validity(completed_digits, mask, size=grid_size)
        total_valid += valid
        total_predictions += total
        
        if valid == total:
            fully_solved += 1
    
    # Calculate rates and standard errors
    pred_rate = total_valid / total_predictions * 100 if total_predictions > 0 else 0
    solve_rate = fully_solved / num_samples * 100
    
    p_pred = total_valid / total_predictions if total_predictions > 0 else 0
    se_pred = np.sqrt(p_pred * (1 - p_pred) / total_predictions) * 100 if total_predictions > 0 else 0
    
    p_solve = fully_solved / num_samples
    se_solve = np.sqrt(p_solve * (1 - p_solve) / num_samples) * 100
    
    return EvaluationResult(
        prediction_validity=pred_rate,
        prediction_se=se_pred,
        solve_rate=solve_rate,
        solve_se=se_solve,
        num_samples=num_samples,
        num_predictions=total_predictions,
        difficulty=num_empty
    )


def evaluate_model_all_difficulties(
    model: torch.nn.Module,
    device: torch.device,
    difficulties: List[int] = [4, 6, 8, 10, 12],
    num_samples: int = 200,
    verbose: bool = True
) -> Dict[int, EvaluationResult]:
    """Evaluate a model across multiple difficulty levels.
    
    Args:
        model: Model to evaluate
        device: Device to run on
        difficulties: List of num_empty values to test
        num_samples: Samples per difficulty
        verbose: Whether to print progress
        
    Returns:
        Dictionary mapping difficulty -> EvaluationResult
    """
    results = {}
    
    for num_empty in difficulties:
        if verbose:
            print(f"  Evaluating {num_empty} empty cells...", end=" ")
        
        result = evaluate_model(model, device, num_empty, num_samples)
        results[num_empty] = result
        
        if verbose:
            print(f"{result.prediction_validity:.1f}% valid ({result.solve_rate:.1f}% solved)")
    
    return results


def compare_models(
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    difficulties: List[int] = [4, 6, 8, 10, 12],
    num_samples: int = 300
) -> Dict[str, Dict[int, EvaluationResult]]:
    """Compare multiple models across all difficulty levels.
    
    Args:
        models: Dictionary mapping model name -> model
        device: Device to run on
        difficulties: Difficulty levels to test
        num_samples: Samples per difficulty per model
        
    Returns:
        Nested dict: model_name -> difficulty -> EvaluationResult
    """
    all_results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        all_results[name] = evaluate_model_all_difficulties(
            model, device, difficulties, num_samples
        )
    
    return all_results


def print_comparison_table(
    results: Dict[str, Dict[int, EvaluationResult]],
    difficulties: List[int] = [4, 6, 8, 10, 12]
) -> None:
    """Print a formatted comparison table.
    
    Args:
        results: Output from compare_models
        difficulties: Difficulty levels to include
    """
    model_names = list(results.keys())
    
    # Header
    print("=" * 100)
    print("MODEL COMPARISON: Prediction Validity Rate")
    print("=" * 100)
    print()
    
    header = f"{'Empty':<8}"
    for name in model_names:
        header += f" {name:<24}"
    print(header)
    print("-" * 100)
    
    # Data rows
    for num_empty in difficulties:
        row = f"{num_empty:<8}"
        for name in model_names:
            r = results[name][num_empty]
            row += f" {r.prediction_validity:>5.1f}% ±{r.prediction_se:.1f}  ({r.solve_rate:>5.1f}% solved) "
        print(row)
    
    # Average row
    print("-" * 100)
    avg_row = f"{'AVG':<8}"
    for name in model_names:
        avg_pred = np.mean([results[name][d].prediction_validity for d in difficulties])
        avg_solve = np.mean([results[name][d].solve_rate for d in difficulties])
        avg_row += f" {avg_pred:>5.1f}%     ({avg_solve:>5.1f}% solved)       "
    print(avg_row)
    print()


def plot_comparison(
    results: Dict[str, Dict[int, EvaluationResult]],
    difficulties: List[int] = [4, 6, 8, 10, 12],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """Create comparison plots with error bars.
    
    Args:
        results: Output from compare_models
        difficulties: Difficulty levels to plot
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    colors = ['#FFB347', '#77DD77', '#779ECB', '#FF6B6B', '#C9B1FF'][:n_models]
    
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    x = np.arange(len(difficulties))
    width = 0.8 / n_models
    
    # Left plot: Prediction validity
    ax = axes[0]
    for i, name in enumerate(model_names):
        vals = [results[name][d].prediction_validity for d in difficulties]
        errs = [results[name][d].prediction_se for d in difficulties]
        offset = (i - n_models/2 + 0.5) * width
        ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
               label=name, color=colors[i], alpha=0.9)
    
    ax.set_ylabel('Valid Predictions %', fontsize=11)
    ax.set_xlabel('Puzzle Difficulty', fontsize=11)
    ax.set_title('Prediction Validity (±1 SE)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{d} empty' for d in difficulties], fontsize=10)
    ax.legend(loc='lower left', fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right plot: Solve rate
    ax2 = axes[1]
    for i, name in enumerate(model_names):
        vals = [results[name][d].solve_rate for d in difficulties]
        errs = [results[name][d].solve_se for d in difficulties]
        offset = (i - n_models/2 + 0.5) * width
        ax2.bar(x + offset, vals, width, yerr=errs, capsize=3,
                label=name, color=colors[i], alpha=0.9)
    
    ax2.set_ylabel('Fully Solved %', fontsize=11)
    ax2.set_xlabel('Puzzle Difficulty', fontsize=11)
    ax2.set_title('Puzzles Fully Solved (all predictions valid)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{d} empty' for d in difficulties], fontsize=10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def get_summary_stats(
    results: Dict[str, Dict[int, EvaluationResult]],
    difficulties: List[int] = [4, 6, 8, 10, 12]
) -> Dict[str, Dict[str, float]]:
    """Get summary statistics for each model.
    
    Args:
        results: Output from compare_models
        difficulties: Difficulty levels to include
        
    Returns:
        Dictionary with summary stats per model
    """
    summary = {}
    
    for name in results.keys():
        pred_vals = [results[name][d].prediction_validity for d in difficulties]
        solve_vals = [results[name][d].solve_rate for d in difficulties]
        
        summary[name] = {
            'avg_prediction_validity': np.mean(pred_vals),
            'std_prediction_validity': np.std(pred_vals),
            'avg_solve_rate': np.mean(solve_vals),
            'std_solve_rate': np.std(solve_vals),
            'min_prediction_validity': np.min(pred_vals),
            'max_prediction_validity': np.max(pred_vals),
        }
    
    return summary


# Convenience function for quick evaluation
def quick_evaluate(
    model: torch.nn.Module,
    device: torch.device,
    num_samples: int = 100
) -> None:
    """Quick evaluation with printed results.
    
    Args:
        model: Model to evaluate
        device: Device to use
        num_samples: Samples per difficulty (lower = faster)
    """
    print("Quick Evaluation")
    print("=" * 50)
    
    results = evaluate_model_all_difficulties(
        model, device, 
        difficulties=[4, 8, 12],
        num_samples=num_samples
    )
    
    avg_pred = np.mean([r.prediction_validity for r in results.values()])
    avg_solve = np.mean([r.solve_rate for r in results.values()])
    
    print()
    print(f"Average Prediction Validity: {avg_pred:.1f}%")
    print(f"Average Solve Rate: {avg_solve:.1f}%")

