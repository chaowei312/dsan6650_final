import numpy as np
import torch
import torch.nn.functional as F

import core
from mcts_node import MCTSNode
from visualizer import LiveSudokuVisualizer


class SudokuMCTS:
    """
    MCTS for single-player Sudoku using a policy + value network.
    """

    def __init__(self, model, c_puct=1.0):
        # store network and exploration constant
        self.model = model
        self.c_puct = c_puct

    def search(
        self,
        root_state,
        num_simulations=200,
        visualizer=None,
        original_puzzle=None,
    ):
        # root node corresponds to the current puzzle state
        root = MCTSNode(root_state)
        # expansion to get root children and value
        self._expand(root)

        # ECR = number of root children
        ecr = len(root.children)
        ent = 0.0

        # compute entropy over root priors
        if root.children:
            pr = np.array([ch.prior for ch in root.children.values()], dtype=np.float32)
            if pr.sum() > 0:
                pr = pr / pr.sum()
                ent = float(-(pr * np.log(pr + 1e-12)).sum())


        if root.children:
            eps = 0.25
            a = 0.3
            noise = np.random.dirichlet([a] * len(root.children))
            for ch, eta in zip(root.children.values(), noise):
                ch.prior = (1 - eps) * ch.prior + eps * eta

        # heatmap for visualizing visit counts per cell
        hm = np.zeros((core.GRID_SIZE, core.GRID_SIZE))

        # main MCTS simulation loop
        for s in range(num_simulations):
            n = root
            path = [n]

            # selection, follow tree using PUCT until leaf
            while n.is_expanded():
                act, n = self._select_child(n)
                path.append(n)

            # expansion, value estimate at leaf
            v = self._expand(n)
            # backup value along the path
            self._backpropagate(path, v)

            if visualizer and s % max(1, num_simulations // 4) == 0 and root.children:
                hm.fill(0.0)
                best_txt = ""
                mx = 0

                # accumulate visit counts into heatmap
                for (idx, d), ch in root.children.items():
                    r, c = divmod(idx, core.GRID_SIZE)
                    vs = ch.visit_count
                    hm[r, c] = vs
                    if vs > mx:
                        mx = vs
                        best_txt = f"R{r}C{c} -> {d}"

                # normalize heatmap to [0,1]
                if mx > 0:
                    hm /= mx

                visualizer.plot_state(
                    board=root_state.reshape(core.GRID_SIZE, core.GRID_SIZE),
                    original_board=(
                        original_puzzle.reshape(core.GRID_SIZE, core.GRID_SIZE)
                        if original_puzzle is not None
                        else None
                    ),
                    policy_heatmap=hm,
                    step_info=(
                        f"Sim {s}/{num_simulations} | "
                        f"Best: {best_txt} | Root Q: {root.q_value:.2f}"
                    ),
                )

        # build AlphaZero-style policy target π from visit counts
        pi = np.zeros((core.TOTAL_CELLS, core.NUM_DIGITS), dtype=np.float32)
        tot = 0

        # fill policy tensor: index by (cell, digit)
        for (idx, d), ch in root.children.items():
            a_idx = idx
            d_idx = d - 1
            pi[a_idx, d_idx] = ch.visit_count
            tot += ch.visit_count

        # normalize to a probability distribution
        if tot > 0:
            pi /= float(tot)

        # pick the move with highest visit count
        best = self._get_best_move(root)
        return best, ecr, ent, pi

    def _select_child(self, node):
        # PUCT selection among children
        best_score = -float("inf")
        best_act = None
        best_ch = None

        # total visits at node (for exploration term)
        tot = max(1, node.visit_count)
        rt = np.sqrt(tot)

        # loop over all legal actions from this node
        for act, ch in node.children.items():
            # exploration bonus
            u = (
                self.c_puct
                * ch.prior
                * rt
                / (1 + ch.visit_count)
            )
            score = ch.q_value + u

            # keep best-scoring child
            if score > best_score:
                best_score = score
                best_act = act
                best_ch = ch

        return best_act, best_ch

    def _expand(self, node):
        # terminal
        if 0 not in node.state:
            if core.is_valid_state(node.state):
                return 1.0
            else:
                return -1.0

        # legal moves from this state
        vm = core.get_valid_moves(node.state)
        if not vm:
            return -1.0

        # MRV focus on cell with fewest options
        f_idx, f_dig = min(vm.items(), key=lambda kv: len(kv[1]))

        # encode board for the network
        st = torch.tensor(
            node.state, dtype=torch.long, device=core.CONFIG["device"]
        ).unsqueeze(0)

        # forward pass through policy and value network
        self.model.eval()
        with torch.no_grad():
            pl, v = self.model(st)


        v = float(v.item())

        # policy logits over all cells/digits
        pl = pl.squeeze(0).cpu()
        p = F.softmax(pl, dim=-1).numpy()

        cand = []
        s = 0.0
        # restrict to top-k digits for 9x9
        top_k = 3 if core.GRID_SIZE == 9 else None

        tmp = []
        # collect (digit, prob) 
        for d in f_dig:
            d_idx = d - 1
            prob = float(p[f_idx, d_idx])
            tmp.append((d, prob))

        # sort digits by probability
        tmp.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            tmp = tmp[:top_k]

        # create child states for chosen digits
        for d, raw in tmp:
            ns = node.state.copy()
            ns[f_idx] = d

            prob = max(raw, 1e-8)
            cand.append(((f_idx, d), ns, prob))
            s += prob

        # no valid child 
        if not cand:
            return -1.0

        if s == 0.0:
            s = 1.0

        # register children under this node with  priors
        for (idx, d), ns, raw in cand:
            pr = raw / s
            ch = MCTSNode(ns, parent=node, move=(idx, d))
            ch.prior = pr
            node.children[(idx, d)] = ch

        return v

    def _backpropagate(self, path, v):
        # update value_sum and visit_count along the path
        for n in reversed(path):
            n.value_sum += v
            n.visit_count += 1

    def _get_best_move(self, root):
        # choose child with the highest visit count
        if not root.children:
            return None
        best_act, _ = max(
            root.children.items(), key=lambda it: it[1].visit_count
        )
        return best_act
