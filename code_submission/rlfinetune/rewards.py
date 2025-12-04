"""
Reward Functions for Sudoku RL

Rule-based rewards - no learned reward model needed!
"""

import torch


def compute_sudoku_reward(puzzle: torch.Tensor, solution: torch.Tensor) -> torch.Tensor:
    """
    Compute reward based on Sudoku constraint violations.
    
    Args:
        puzzle: [B, 16] - puzzle with clues (1=empty, 2-5=digits)
        solution: [B, 16] - proposed solution
        
    Returns:
        reward: [B] - reward in [0, 1] where 1 = valid solution (0 violations)
    """
    B = solution.shape[0]
    board = solution.view(B, 4, 4)
    device = solution.device
    
    violations = torch.zeros(B, device=device)
    
    # Count row and column violations
    for i in range(4):
        row, col = board[:, i, :], board[:, :, i]
        for d in range(2, 6):  # Digits 1-4 (encoded as 2-5)
            violations += ((row == d).sum(dim=1) > 1).float()
            violations += ((col == d).sum(dim=1) > 1).float()
    
    # Count box violations (2x2 boxes)
    for br in range(2):
        for bc in range(2):
            box = board[:, br*2:(br+1)*2, bc*2:(bc+1)*2].reshape(B, 4)
            for d in range(2, 6):
                violations += ((box == d).sum(dim=1) > 1).float()
    
    # Check clue preservation
    clue_mask = (puzzle > 1)
    clues_preserved = ((solution == puzzle) | ~clue_mask).all(dim=1).float()
    
    # Reward: 1 for valid (0 violations), 0 for max violations (24)
    # Max violations = 24 (each row/col/box can have at most 2 digits appearing >1)
    reward = (1.0 - violations / 24.0) * clues_preserved
    return reward.clamp(0, 1)


def compute_dense_reward(
    puzzle: torch.Tensor, 
    solution: torch.Tensor,
    ground_truth: torch.Tensor,
) -> torch.Tensor:
    """
    Compute dense reward based on cell-level accuracy.
    
    Args:
        puzzle: [B, 16] - puzzle with clues
        solution: [B, 16] - proposed solution
        ground_truth: [B, 16] - correct solution
        
    Returns:
        reward: [B] - reward based on fraction of correct cells
    """
    # Only count empty cells (where puzzle == 1)
    empty_mask = (puzzle == 1)
    
    # Count correct predictions on empty cells
    correct = ((solution == ground_truth) & empty_mask).float().sum(dim=1)
    total = empty_mask.float().sum(dim=1).clamp(min=1)
    
    # Scale to [-1, +1]: all wrong = -1, all correct = +1
    accuracy = correct / total
    reward = 2 * accuracy - 1
    
    return reward


def compute_hybrid_reward(
    puzzle: torch.Tensor,
    solution: torch.Tensor,
    ground_truth: torch.Tensor,
    constraint_weight: float = 0.5,
) -> torch.Tensor:
    """
    Hybrid reward combining constraint violations and accuracy.
    
    Args:
        puzzle: [B, 16] - puzzle with clues
        solution: [B, 16] - proposed solution
        ground_truth: [B, 16] - correct solution
        constraint_weight: weight for constraint-based reward (0-1)
        
    Returns:
        reward: [B] - combined reward
    """
    constraint_reward = compute_sudoku_reward(puzzle, solution)
    accuracy_reward = compute_dense_reward(puzzle, solution, ground_truth)
    
    return constraint_weight * constraint_reward + (1 - constraint_weight) * accuracy_reward

