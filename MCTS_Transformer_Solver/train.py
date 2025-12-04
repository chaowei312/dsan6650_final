import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

import core
from sudoku_net import SudokuNet
from visualizer import LiveSudokuVisualizer
from online_solver import OnlineLearningSolver


def train_model(dim=9, n_puzzle=100):
    """
    Train the model on `n_puzzle` puzzles 
    """

    core.GRID_SIZE = dim
    core.BLOCK_SIZE = 2 if dim == 4 else 3
    core.TOTAL_CELLS = core.GRID_SIZE * core.GRID_SIZE
    core.NUM_DIGITS = core.GRID_SIZE
    # get params from core
    model = SudokuNet().to(core.CONFIG["device"])

    viz = LiveSudokuVisualizer()
    solver = OnlineLearningSolver(
        model,
        viz,
        lr=0.001,
        buffer_capacity=20000,
        batch_size=256,
        updates_per_episode=20,
        value_loss_weight=1.0,
    )

    ep_res = []
    ecr_ep = []
    ent_ep = []

    for ep in tqdm(range(1, n_puzzle + 1), desc="Episode Progress"):
        viz.reset_frames()

        if core.GRID_SIZE == 4:
            min_g, max_g = 4, 8
            max_steps = 40
            sims = 60
        else:
            min_g, max_g = 30, 60
            max_steps = 60
            sims = 200

        sol = core.generate_random_solved_board()
        p = core.mask_puzzle(sol, min_given=min_g, max_given=max_g)

        ok, avg_ecr, avg_ent, steps = solver.solve(
            p,
            max_steps=max_steps,
            simulations_per_move=sims,
        )

        ep_res.append(1 if ok else 0)
        ecr_ep.append(avg_ecr)
        ent_ep.append(avg_ent)

        time.sleep(0.05)

    print("Training complete.")
    # save frames as gif
    if viz.frames:
        gif_name = f"sudoku{dim}x{dim}_solve.gif"
        imageio.mimsave(gif_name, viz.frames, fps=5)
        print(f"Saved solving process GIF to {gif_name}")
    else:
        print("No frames recorded for GIF.")


    res_arr = np.array(ep_res, dtype=np.float32)
    sm_win = []
    w = 50
    for i in range(len(res_arr)):
        st = max(0, i - w + 1)
        win = res_arr[st:i+1]
        sm_win.append(win.mean())
    sm_win = np.array(sm_win)

    eps = np.arange(1, len(ecr_ep) + 1)
    # take moving average across 50 episodes and plot
    plt.figure(figsize=(15, 4))
    w = 50
    sm_ecr = core.moving_average(ecr_ep, window=w)
    sm_ent = core.moving_average(ent_ep, window=w)

    ecr_c = np.polyfit(eps, sm_ecr, deg=1)
    ecr_tr = np.polyval(ecr_c, eps)
    ent_c = np.polyfit(eps, sm_ent, deg=1)
    ent_tr = np.polyval(ent_c, eps)

    plt.subplot(1, 3, 1)
    plt.plot(eps, sm_ecr, marker="o")
    plt.plot(eps, ecr_tr, color="red", linewidth=2, label="Trend")
    plt.xlabel("Episode")
    plt.ylabel("Avg ECR (root children)")
    plt.title(f"Smoothed ECR per Episode (window={w})")
    plt.grid(True, linestyle="--", alpha=0.5)


    plt.subplot(1, 3, 2)
    plt.plot(eps, sm_ent, marker="o")
    plt.plot(eps, ent_tr, color="red", linewidth=2, label="Trend")
    plt.xlabel("Episode")
    plt.ylabel("Avg Root Policy Entropy")
    plt.title(f"Smoothed Policy Entropy (window={w})")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 3, 3)
    plt.plot(eps, sm_win, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Winrate (moving avg, window=10)")
    plt.ylim(0.0, 1.0)
    plt.title("Smoothed Winrate over Episodes")
    plt.grid(True, linestyle="--", alpha=0.5)


    plt.tight_layout()
    metrics_png = f"sudoku{dim}x{dim}_metrics.png"
    plt.savefig(metrics_png)
    plt.show()


    wins = sum(ep_res)
    tot = len(ep_res)
    winrate = wins / tot if tot > 0 else 0.0
    print(f"Winrate over {tot} puzzles: {wins}/{tot} = {winrate*100:.2f}%")

    model_path = f"sudoku_model_{dim}x{dim}.pt"
    torch.save(model.state_dict(), model_path)



if __name__ == "__main__":
    train_model(dim=9, n_puzzle=1000)
