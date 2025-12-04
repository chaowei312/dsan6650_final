import numpy as np


class ReplayBuffer:
    """Keep a record of (s,v,p)"""
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.s = []
        self.v = []
        self.p = []

    def __len__(self):
        return len(self.s)

    def add_episode(self, states, policy_targets, target_value):
        assert len(states) == len(policy_targets)
        # maintain a queue
        for st, pi in zip(states, policy_targets):
            if len(self.s) >= self.capacity:
                self.s.pop(0)
                self.v.pop(0)
                self.p.pop(0)
            self.s.append(st.copy())
            self.v.append(target_value)
            self.p.append(pi.astype(np.float32))

    def sample_batch(self, batch_size):
        batch_size = min(batch_size, len(self.s))
        idx = np.random.choice(len(self.s), size=batch_size, replace=False)
        bs = np.stack([self.s[i] for i in idx], axis=0)
        bv = np.array([self.v[i] for i in idx], dtype=np.float32)
        bp = np.stack([self.p[i] for i in idx], axis=0)
        return bs, bp, bv.reshape(-1, 1)
