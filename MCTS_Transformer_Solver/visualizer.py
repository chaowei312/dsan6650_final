import numpy as np
import pygame

import core


class LiveSudokuVisualizer:
    """
    visualizer for Sudoku boards using pygame
    """

    def __init__(self, cell_size=40):
        self.cell_size = cell_size
        self.width = core.GRID_SIZE * cell_size
        self.height = core.GRID_SIZE * cell_size

        pygame.init()
        pygame.font.init()

        self.surface = pygame.Surface((self.width, self.height))
        self.font = pygame.font.SysFont("arial", int(self.cell_size * 0.6))
        self.frames = []

    def reset_frames(self):
        self.frames = []

    def _render_frame(self, board, original_board, policy_heatmap, current_focus):
        self.surface.fill((255, 255, 255))

        if policy_heatmap is not None:
            hm = policy_heatmap
            if hm.max() > 0:
                hm = hm / hm.max()
            for r in range(core.GRID_SIZE):
                for c in range(core.GRID_SIZE):
                    v = float(hm[r, c])
                    rc = int(255 - 205 * v)
                    gc = int(255 - 205 * v)
                    bc = 255
                    col = (rc, gc, bc)
                    x = c * self.cell_size
                    y = r * self.cell_size
                    pygame.draw.rect(
                        self.surface,
                        col,
                        pygame.Rect(x, y, self.cell_size, self.cell_size),
                    )
        # draw the grid
        for r in range(core.GRID_SIZE + 1):
            y = r * self.cell_size
            pygame.draw.line(self.surface, (0, 0, 0), (0, y), (self.width, y), width=1)
        for c in range(core.GRID_SIZE + 1):
            x = c * self.cell_size
            pygame.draw.line(self.surface, (0, 0, 0), (x, 0), (x, self.height), width=1)

        for r in range(0, core.GRID_SIZE + 1, core.BLOCK_SIZE):
            y = r * self.cell_size
            pygame.draw.line(self.surface, (0, 0, 0), (0, y), (self.width, y), width=3)
        for c in range(0, core.GRID_SIZE + 1, core.BLOCK_SIZE):
            x = c * self.cell_size
            pygame.draw.line(self.surface, (0, 0, 0), (x, 0), (x, self.height), width=3)

        for r in range(core.GRID_SIZE):
            for c in range(core.GRID_SIZE):
                v = int(board[r, c])
                x = c * self.cell_size
                y = r * self.cell_size

                if current_focus == (r, c):
                    pygame.draw.rect(
                        self.surface,
                        (255, 0, 0),
                        pygame.Rect(x, y, self.cell_size, self.cell_size),
                        width=3,
                    )

                if v == 0:
                    continue

                col = (0, 0, 0)
                if original_board is not None and int(original_board[r, c]) == 0:
                    col = (0, 0, 255)

                t_surf = self.font.render(str(v), True, col)
                t_rect = t_surf.get_rect()
                t_rect.center = (x + self.cell_size // 2, y + self.cell_size // 2)
                self.surface.blit(t_surf, t_rect)

        frame = pygame.surfarray.array3d(self.surface)
        frame = np.transpose(frame, (1, 0, 2))
        return frame

    def plot_state(
        self,
        board,
        original_board=None,
        policy_heatmap=None,
        current_focus=None,
        step_info=""
    ):
        f = self._render_frame(
            board,
            original_board,
            policy_heatmap,
            current_focus,
        )
        self.frames.append(f)
