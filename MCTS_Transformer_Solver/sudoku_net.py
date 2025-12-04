# sudoku_net.py
import torch
import torch.nn as nn

import core


class SudokuNet(nn.Module):
    """
    Transformer-based policy and value network
    """

    def __init__(self):
        super().__init__()
        
        self.value_embedding = nn.Embedding(core.NUM_DIGITS + 1, core.CONFIG["d_model"])
        self.pos_embedding = nn.Embedding(core.TOTAL_CELLS, core.CONFIG["d_model"])

        enc = nn.TransformerEncoderLayer(
            d_model=core.CONFIG["d_model"],
            nhead=core.CONFIG["nhead"],
            dim_feedforward=core.CONFIG["dim_feedforward"],
            dropout=core.CONFIG["dropout"],
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc, num_layers=core.CONFIG["num_layers"]
        )

        self.policy_head = nn.Sequential(
            nn.Linear(core.CONFIG["d_model"], 128),
            nn.ReLU(),
            nn.Linear(128, core.NUM_DIGITS),
        )

        self.value_head = nn.Sequential(
            nn.Linear(core.CONFIG["d_model"], 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        b, t = x.shape
        dev = x.device

        pos = torch.arange(0, core.TOTAL_CELLS, device=dev).unsqueeze(0).repeat(b, 1)

        # add the value and position embedding together to feed the transformer
        x_emb = self.value_embedding(x) + self.pos_embedding(pos)

        feat = self.transformer(x_emb)

        # get policy head and value head
        pol = self.policy_head(feat)

        g = feat.mean(dim=1)
        val = self.value_head(g)

        return pol, val
