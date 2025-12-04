import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import core
from replay_buffer import ReplayBuffer
from visualizer import LiveSudokuVisualizer
from mcts import SudokuMCTS
from sudoku_net import SudokuNet


class OnlineLearningSolver:
    """
    Sudoku solver that mimics AlphaZero-style self-play + training.
    """

    def __init__(
        self,
        model,
        viz,
        lr=0.0005,
        buffer_capacity=20000,
        batch_size=128,
        updates_per_episode=20,
        value_loss_weight=1.0,
    ):
        # network + visualization handle
        self.model = model
        self.viz = viz

        # optimizer and loss
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.mse = nn.MSELoss()

        self.bs = batch_size
        self.upd = updates_per_episode
        self.v_w = value_loss_weight

        self.buf = ReplayBuffer(capacity=buffer_capacity)

        self.model.to(core.CONFIG["device"])
        self.model.eval()

    def _train_from_buffer(self):
        # not enough data to form a full batch
        if len(self.buf) < self.bs:
            print(f"Buffer too small ({len(self.buf)}), skipping update")
            return

        # switch to training mode
        self.model.train()
        last = None

        # multiple gradient updates per episode
        for _ in range(self.upd):
            st_np, pi_np, v_np = self.buf.sample_batch(self.bs)

            # states to long tensor
            st = torch.tensor(
                st_np,
                dtype=torch.long,
                device=core.CONFIG["device"],
            )
            # policy targets
            pi_t = torch.tensor(
                pi_np,
                dtype=torch.float32,
                device=core.CONFIG["device"],
            )
            # value targets
            v_t = torch.tensor(
                v_np,
                dtype=torch.float32,
                device=core.CONFIG["device"],
            )

            self.opt.zero_grad()
            logits, v_pred = self.model(st)

            # flatten 
            b = logits.shape[0]
            log_flat = logits.view(b, -1)
            pi_flat = pi_t.view(b, -1)

            # log-probabilities under current policy
            log_p = F.log_softmax(log_flat, dim=-1)

            # cross-entropy between target π and model
            p_loss = -(pi_flat * log_p).sum(dim=-1).mean()

            # MSE between predicted and target value
            v_loss = self.mse(v_pred, v_t)

            # combined loss
            loss = p_loss + self.v_w * v_loss
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()
            last = loss.item()

        self.model.eval()

    def solve(self, start_puzzle, max_steps=100, simulations_per_move=200):
        orig = start_puzzle.copy()
        cur = start_puzzle.copy()
        mcts = SudokuMCTS(self.model)

        ep_s = []
        ep_pi = []

        ecr = []
        ent = []
        stuck = False

        # play up to max_steps moves
        for step in range(1, max_steps + 1):
            if core.is_solved(cur):
                break

            if not core.is_valid_state(cur):
                print("Board state invalid")
                stuck = True
                break

            # run MCTS from current board
            best, e_step, h_step, pi = mcts.search(
                cur,
                num_simulations=simulations_per_move,
                visualizer=self.viz,
                original_puzzle=orig,
            )

            # no moves available
            if best is None:
                print("MCTS returned no moves.")
                stuck = True
                break

            ep_s.append(cur.copy())
            ep_pi.append(pi.copy())

            ecr.append(float(e_step))
            ent.append(float(h_step))

            idx, val = best
            r, c = divmod(idx, core.GRID_SIZE)

            # apply chosen move to board
            cur[idx] = val

            # visualize current board + move
            self.viz.plot_state(
                cur.reshape(core.GRID_SIZE, core.GRID_SIZE),
                original_board=orig.reshape(core.GRID_SIZE, core.GRID_SIZE),
                current_focus=(r, c),
                step_info=f"Step {step}: Placed {val} at ({r},{c})",
            )

            if core.is_solved(cur):
                break

        # final puzzle outcome
        solved = core.is_solved(cur) and not stuck
        steps = len(ep_s)

        # only train if at least one decision was made
        if ep_s:
            tv = 1.0 if solved else -1.0
            self.buf.add_episode(ep_s, ep_pi, tv)
            self._train_from_buffer()
        else:
            print("Episode produced no decision states")

        # aggregate episode-level metrics
        if ecr:
            avg_ecr = float(np.mean(ecr))
        else:
            avg_ecr = 0.0

        if ent:
            avg_ent = float(np.mean(ent))
        else:
            avg_ecr = 0.0

        return solved, avg_ecr, avg_ent, steps
