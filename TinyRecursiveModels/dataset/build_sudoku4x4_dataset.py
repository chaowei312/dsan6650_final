"""
4x4 Mini Sudoku Dataset Generator

Generates 4x4 Sudoku puzzles with configurable difficulty.
No external dataset needed - generates valid puzzles programmatically.

Grid layout (2x2 boxes):
┌───┬───┐
│0 1│2 3│
│4 5│6 7│
├───┼───┤
│8 9│A B│
│C D│E F│
└───┴───┘
"""
from typing import Optional, List, Tuple
import os
import json
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

try:
    from argdantic import ArgParser
    from pydantic import BaseModel
    from common import PuzzleDatasetMetadata
    HAS_ARGDANTIC = True
except ImportError:
    HAS_ARGDANTIC = False
    BaseModel = object


# ============== Sudoku 4x4 Logic ==============

def get_box_idx(row: int, col: int) -> int:
    """Get which 2x2 box a cell belongs to."""
    return (row // 2) * 2 + (col // 2)


def is_valid_placement(grid: np.ndarray, row: int, col: int, num: int) -> bool:
    """Check if placing num at (row, col) is valid."""
    # Check row
    if num in grid[row, :]:
        return False
    # Check column
    if num in grid[:, col]:
        return False
    # Check 2x2 box
    box_row, box_col = 2 * (row // 2), 2 * (col // 2)
    if num in grid[box_row:box_row+2, box_col:box_col+2]:
        return False
    return True


def solve_sudoku4x4(grid: np.ndarray, find_all: bool = False) -> List[np.ndarray]:
    """
    Solve 4x4 Sudoku using backtracking.
    
    Args:
        grid: 4x4 numpy array (0 = empty)
        find_all: If True, find all solutions (for uniqueness check)
    
    Returns:
        List of solution grids
    """
    grid = grid.copy()
    solutions = []
    
    def solve(g):
        # Find empty cell
        empty_cells = np.where(g == 0)
        if len(empty_cells[0]) == 0:
            solutions.append(g.copy())
            return not find_all  # Stop if we only need one
        
        row, col = empty_cells[0][0], empty_cells[1][0]
        
        for num in range(1, 5):
            if is_valid_placement(g, row, col, num):
                g[row, col] = num
                if solve(g):
                    return True
                g[row, col] = 0
        
        return False
    
    solve(grid)
    return solutions


def generate_complete_grid() -> np.ndarray:
    """Generate a random valid complete 4x4 Sudoku grid."""
    grid = np.zeros((4, 4), dtype=np.int32)
    
    def fill(g):
        empty_cells = np.where(g == 0)
        if len(empty_cells[0]) == 0:
            return True
        
        row, col = empty_cells[0][0], empty_cells[1][0]
        nums = np.random.permutation(np.arange(1, 5))
        
        for num in nums:
            if is_valid_placement(g, row, col, num):
                g[row, col] = num
                if fill(g):
                    return True
                g[row, col] = 0
        return False
    
    fill(grid)
    return grid


def create_puzzle(solution: np.ndarray, num_clues: int, ensure_unique: bool = True) -> Optional[np.ndarray]:
    """
    Create a puzzle by removing cells from a solution.
    
    Args:
        solution: Complete 4x4 grid
        num_clues: Number of cells to keep revealed
        ensure_unique: If True, verify puzzle has unique solution
    
    Returns:
        Puzzle grid (0 = empty) or None if couldn't create unique puzzle
    """
    cells = list(range(16))
    np.random.shuffle(cells)
    
    puzzle = solution.copy()
    removed = []
    
    for cell in cells:
        if 16 - len(removed) <= num_clues:
            break
            
        row, col = cell // 4, cell % 4
        old_val = puzzle[row, col]
        puzzle[row, col] = 0
        
        if ensure_unique:
            solutions = solve_sudoku4x4(puzzle, find_all=True)
            if len(solutions) != 1:
                # Multiple solutions or no solution - restore
                puzzle[row, col] = old_val
            else:
                removed.append(cell)
        else:
            removed.append(cell)
    
    return puzzle


def augment_sudoku4x4(board: np.ndarray, solution: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment a 4x4 Sudoku puzzle while preserving validity.
    - Permute digits 1-4
    - Swap rows within bands (row 0↔1, row 2↔3)
    - Swap columns within stacks (col 0↔1, col 2↔3)
    - Swap bands (rows 0-1 ↔ rows 2-3)
    - Swap stacks (cols 0-1 ↔ cols 2-3)
    - Transpose
    """
    # Digit permutation (0 stays 0)
    digit_map = np.concatenate([[0], np.random.permutation(np.arange(1, 5))])
    
    # Row/col permutations within 2x2 bands/stacks
    row_perm = np.concatenate([
        np.random.permutation([0, 1]) if np.random.rand() < 0.5 else [0, 1],
        np.random.permutation([2, 3]) if np.random.rand() < 0.5 else [2, 3]
    ])
    col_perm = np.concatenate([
        np.random.permutation([0, 1]) if np.random.rand() < 0.5 else [0, 1],
        np.random.permutation([2, 3]) if np.random.rand() < 0.5 else [2, 3]
    ])
    
    # Swap bands/stacks
    if np.random.rand() < 0.5:
        row_perm = np.array([row_perm[2], row_perm[3], row_perm[0], row_perm[1]])
    if np.random.rand() < 0.5:
        col_perm = np.array([col_perm[2], col_perm[3], col_perm[0], col_perm[1]])
    
    # Transpose
    transpose = np.random.rand() < 0.5
    
    def transform(x):
        x = x.copy()
        if transpose:
            x = x.T
        x = x[row_perm, :][:, col_perm]
        return digit_map[x]
    
    return transform(board), transform(solution)


# ============== Dataset Generation ==============

@dataclass
class Sudoku4x4Config:
    output_dir: str = "data/sudoku4x4"
    num_train: int = 10000
    num_test: int = 1000
    min_clues: int = 4
    max_clues: int = 10
    num_aug: int = 0
    ensure_unique: bool = True
    # RL mode: generate trajectories from different starting points
    rl_mode: bool = False
    steps_per_puzzle: int = 5  # For RL: generate this many starting points per puzzle


def generate_dataset(config: Sudoku4x4Config, set_name: str, num_samples: int):
    """Generate a dataset of 4x4 Sudoku puzzles."""
    
    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [0],
        "group_indices": [0],
    }
    
    # For RL mode, also store intermediate states
    if config.rl_mode:
        results["num_filled"] = []  # Track how many cells are filled
    
    puzzle_id = 0
    example_id = 0
    
    pbar = tqdm(total=num_samples, desc=f"Generating {set_name}")
    
    while puzzle_id < num_samples:
        # Generate complete solution
        solution = generate_complete_grid()
        
        # Generate puzzle with random difficulty
        num_clues = np.random.randint(config.min_clues, config.max_clues + 1)
        puzzle = create_puzzle(solution, num_clues, ensure_unique=config.ensure_unique)
        
        if puzzle is None:
            continue  # Retry if couldn't create valid puzzle
        
        # Number of augmentations
        num_aug = config.num_aug if set_name == "train" else 0
        
        for aug_idx in range(1 + num_aug):
            if aug_idx == 0:
                inp, out = puzzle, solution
            else:
                inp, out = augment_sudoku4x4(puzzle, solution)
            
            if config.rl_mode:
                # Generate multiple starting points for RL
                filled_cells = np.where(inp.flatten() > 0)[0]
                empty_cells = np.where(inp.flatten() == 0)[0]
                
                for step in range(config.steps_per_puzzle):
                    # Create state with some cells filled from solution
                    if step == 0:
                        state = inp.copy()
                    else:
                        # Fill random subset of empty cells
                        num_to_fill = np.random.randint(0, len(empty_cells))
                        cells_to_fill = np.random.choice(empty_cells, num_to_fill, replace=False)
                        state = inp.copy().flatten()
                        state[cells_to_fill] = out.flatten()[cells_to_fill]
                        state = state.reshape(4, 4)
                    
                    results["inputs"].append(state)
                    results["labels"].append(out)
                    results["num_filled"].append(np.sum(state > 0))
                    example_id += 1
                    
                results["puzzle_indices"].append(example_id)
                results["puzzle_identifiers"].append(0)
                puzzle_id += 1
            else:
                # Standard mode: one input-output pair
                results["inputs"].append(inp)
                results["labels"].append(out)
                example_id += 1
                puzzle_id += 1
                
                results["puzzle_indices"].append(example_id)
                results["puzzle_identifiers"].append(0)
        
        results["group_indices"].append(puzzle_id)
        pbar.update(1 + num_aug if not config.rl_mode else 1)
    
    pbar.close()
    
    # Convert to numpy
    def to_numpy(arr_list):
        arr = np.stack([a.flatten() for a in arr_list])
        return arr + 1  # Shift: 0->1 (blank becomes 1, digits 1-4 become 2-5)
    
    final_results = {
        "inputs": to_numpy(results["inputs"]),
        "labels": to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }
    
    if config.rl_mode:
        final_results["num_filled"] = np.array(results["num_filled"], dtype=np.int32)
    
    # Metadata
    metadata_dict = {
        "seq_len": 16,
        "vocab_size": 6,  # PAD(0) + blank(1) + digits 1-4 (2-5)
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 1,
        "num_puzzle_identifiers": 1,
        "total_groups": len(results["group_indices"]) - 1,
        "mean_puzzle_examples": 1 if not config.rl_mode else config.steps_per_puzzle,
        "total_puzzles": len(results["group_indices"]) - 1,
        "sets": ["all"]
    }
    
    # Save
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata_dict, f, indent=2)
    
    for k, v in final_results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    
    # Save identifiers mapping
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    
    print(f"Saved {set_name}: {final_results['inputs'].shape[0]} examples")
    return final_results


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate 4x4 Sudoku dataset")
    parser.add_argument("--output_dir", type=str, default="data/sudoku4x4")
    parser.add_argument("--num_train", type=int, default=10000)
    parser.add_argument("--num_test", type=int, default=1000)
    parser.add_argument("--min_clues", type=int, default=4)
    parser.add_argument("--max_clues", type=int, default=10)
    parser.add_argument("--num_aug", type=int, default=0)
    parser.add_argument("--no_unique", action="store_true", help="Don't ensure unique solutions")
    parser.add_argument("--rl_mode", action="store_true", help="Generate RL trajectories")
    parser.add_argument("--steps_per_puzzle", type=int, default=5)
    
    args = parser.parse_args()
    
    config = Sudoku4x4Config(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_test=args.num_test,
        min_clues=args.min_clues,
        max_clues=args.max_clues,
        num_aug=args.num_aug,
        ensure_unique=not args.no_unique,
        rl_mode=args.rl_mode,
        steps_per_puzzle=args.steps_per_puzzle,
    )
    
    print(f"Config: {config}")
    print(f"Grid size: 4x4 (16 cells)")
    print(f"Clues range: {config.min_clues}-{config.max_clues}")
    
    generate_dataset(config, "train", config.num_train)
    generate_dataset(config, "test", config.num_test)
    
    print(f"\nDataset saved to {config.output_dir}")


if __name__ == "__main__":
    main()

