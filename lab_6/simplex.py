import numpy as np

from lab_6.visual_tools import visualize_with_matrix


class SimplexInteractive:
    def __init__(self):
        self.cols = ["x1", "x2", "s1", "s2", "s3", "s4", "a1", "Res"]
        self.row_names = ["Eq1", "Eq2", "Eq3", "Eq4", "F-obj", "W-phase1"]

        # Initial matrix setup
        data = [
            [5.0, -2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0],
            [1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0],
            [-3.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0],
            [2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0],
            [-7.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
        ]
        self.table = np.array(data, dtype=float)
        self.snapshots = [self.table.copy()]

    def _pivot_step(self, row_idx):
        obj_row = self.table[row_idx, :-1]
        p_col = np.argmin(obj_row)
        if obj_row[p_col] >= 0:
            return False

        ratios = []
        for i in range(4):
            val = self.table[i, p_col]
            ratios.append(self.table[i, -1] / val if val > 0 else np.inf)

        p_row = np.argmin(ratios)
        if ratios[p_row] == np.inf:
            return False

        # Danzig's transformation
        self.table[p_row] /= self.table[p_row, p_col]
        for r in range(len(self.table)):
            if r != p_row:
                self.table[r] -= self.table[r, p_col] * self.table[p_row]

        self.snapshots.append(self.table.copy())
        return True

    def calculate(self):
        while self._pivot_step(5):
            pass  # Phase 1
        while self._pivot_step(4):
            pass  # Phase 2
        return self.snapshots


if __name__ == "__main__":
    engine = SimplexInteractive()
    snaps = engine.calculate()
    visualize_with_matrix(snaps, engine.cols, engine.row_names)
