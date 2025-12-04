import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from tqdm import tqdm
import pygame

# core settings + helpers for Sudoku and training
# global default settings , can be overiiden
GRID_SIZE = 9
BLOCK_SIZE = 3
TOTAL_CELLS = GRID_SIZE * GRID_SIZE
NUM_DIGITS = 9 

# model config
CONFIG = {
    "d_model": 512,
    "nhead": 8,
    "num_layers": 4,
    "dim_feedforward": 1024,
    "dropout": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

print("CUDA available:", torch.cuda.is_available())


def is_valid_state(b_flat):
    g = b_flat.reshape(GRID_SIZE, GRID_SIZE)

    def has_dup(a):
        v = a[a > 0] # ignore zeros
        return len(v) != len(np.unique(v))

    for i in range(GRID_SIZE):
        if has_dup(g[i, :]) or has_dup(g[:, i]):
            return False

    for r in range(0, GRID_SIZE, BLOCK_SIZE): # iterate over each row
        for c in range(0, GRID_SIZE, BLOCK_SIZE): # column
            blk = g[r:r + BLOCK_SIZE, c:c + BLOCK_SIZE].flatten()
            if has_dup(blk):
                return False

    return True


def is_solved(b_flat):
    if 0 in b_flat: # check for empty cells
        return False
    return is_valid_state(b_flat)


def get_valid_moves(b_flat):
    g = b_flat.reshape(GRID_SIZE, GRID_SIZE)
    v = {}

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            idx = r * GRID_SIZE + c
            if g[r, c] != 0:    # skip filled cells
                continue

            used = set(g[r, :]) | set(g[:, c])  # row + column digits

            br = (r // BLOCK_SIZE) * BLOCK_SIZE
            bc = (c // BLOCK_SIZE) * BLOCK_SIZE
            used |= set(g[br:br + BLOCK_SIZE, bc:bc + BLOCK_SIZE].flatten())

            opts = [x for x in range(1, NUM_DIGITS + 1) if x not in used]
            if opts:
                v[idx] = opts

    return v


def base_solved_board():
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            g[r, c] = ((r * BLOCK_SIZE + r // BLOCK_SIZE + c) % GRID_SIZE) + 1
    return g.flatten()


def permute_rows_within_bands(g):
    g = g.copy()
    for s in range(0, GRID_SIZE, BLOCK_SIZE):
        rows = np.arange(s, s + BLOCK_SIZE)
        np.random.shuffle(rows) # shuffle rows within band
        g[s:s + BLOCK_SIZE, :] = g[rows, :]
    return g


def permute_cols_within_stacks(g):
    g = g.copy()
    for s in range(0, GRID_SIZE, BLOCK_SIZE): 
        cols = np.arange(s, s + BLOCK_SIZE)
        np.random.shuffle(cols) # shuffle columns within stack
        g[:, s:s + BLOCK_SIZE] = g[:, cols]
    return g


def permute_bands(g):
    g = g.copy()
    n = GRID_SIZE // BLOCK_SIZE
    b = np.arange(n)
    np.random.shuffle(b)

    new_g = np.zeros_like(g)
    for i, bi in enumerate(b):
        src = slice(bi * BLOCK_SIZE, (bi + 1) * BLOCK_SIZE)
        dst = slice(i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE)
        new_g[dst, :] = g[src, :]
    return new_g


def permute_stacks(g):
    g = g.copy()
    n = GRID_SIZE // BLOCK_SIZE
    s = np.arange(n)
    np.random.shuffle(s) # new order of column stacks

    new_g = np.zeros_like(g)
    for i, si in enumerate(s):
        src = slice(si * BLOCK_SIZE, (si + 1) * BLOCK_SIZE)
        dst = slice(i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE)
        new_g[:, dst] = g[:, src]
    return new_g


def permute_digits(g):
    g = g.copy()
    m = np.arange(NUM_DIGITS + 1, dtype=int)  
    p = np.arange(1, NUM_DIGITS + 1, dtype=int)
    np.random.shuffle(p) # random permutation
    m[1:] = p
    g = m[g] # relabel digits
    return g


def generate_random_solved_board():
    b = base_solved_board()
    g = b.reshape(GRID_SIZE, GRID_SIZE)

    g = permute_rows_within_bands(g)
    g = permute_cols_within_stacks(g)
    g = permute_bands(g)
    g = permute_stacks(g)
    g = permute_digits(g)

    return g.flatten()


def mask_puzzle(b_flat, min_given, max_given):
    assert 0 <= min_given <= TOTAL_CELLS
    assert 0 < max_given <= TOTAL_CELLS
    assert min_given <= max_given

    p = b_flat.copy()

    n = np.random.randint(min_given, max_given + 1) 
    n = min(n, TOTAL_CELLS)

    idx = np.arange(TOTAL_CELLS)
    np.random.shuffle(idx) # random order of cells
    keep = idx[:n] # cells to keep as clues

    m = np.zeros(TOTAL_CELLS, dtype=bool)
    m[keep] = True

    p[~m] = 0 # mask everything else
    assert is_valid_state(p), "Masked puzzle is invalid!"
    return p


def moving_average(x, window=10):
    a = np.array(x, dtype=np.float32)
    s = []
    for i in range(len(a)):
        st = max(0, i - window + 1) 
        w = a[st:i+1]
        s.append(w.mean())
    return np.array(s, dtype=np.float32)
