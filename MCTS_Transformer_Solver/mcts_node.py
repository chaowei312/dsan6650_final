import numpy as np


class MCTSNode:
    def __init__(self, board_state, parent=None, move=None):
        # MCTS nodes hold the following info:
        self.state = board_state
        self.parent = parent
        self.move = move

        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 0.0

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        return len(self.children) > 0
