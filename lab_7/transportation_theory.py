import numpy as np
import pandas as pd
from lab_7.visual_tools import visualize_transport


class TransportationInteractive:
    def __init__(self, costs, supply, demand):
        self.costs = np.array(costs, dtype=float)
        self.supply = np.array(supply, dtype=float)
        self.demand = np.array(demand, dtype=float)
        self.m, self.n = self.costs.shape

        self.x = np.zeros((self.m, self.n))
        self.snapshots = []

    def _save_snapshot(self, message, u=None, v=None, deltas=None):
        """Сохраняет текущее состояние для анимации"""
        self.snapshots.append(
            {
                "x": self.x.copy(),
                "u": u.copy() if u is not None else np.full(self.m, np.nan),
                "v": v.copy() if v is not None else np.full(self.n, np.nan),
                "deltas": (
                    deltas.copy() if deltas is not None else np.zeros((self.m, self.n))
                ),
                "msg": message,
                "total_cost": np.sum(self.x * self.costs),
            }
        )

    def run_nw_corner(self):
        """Фаза 1: Метод северо-западного угла"""
        s, d = self.supply.copy(), self.demand.copy()
        i, j = 0, 0
        while i < self.m and j < self.n:
            val = min(s[i], d[j])
            self.x[i, j] = val
            s[i] -= val
            d[j] -= val
            self._save_snapshot(f"NW Corner: Заполнили ({i+1},{j+1})")
            if s[i] == 0:
                i += 1
            else:
                j += 1

    def calculate_potentials(self):
        """Расчет u_i и v_j (u_i + v_j = c_ij для базисных клеток)"""
        u, v = np.full(self.m, np.nan), np.full(self.n, np.nan)
        u[0] = 0

        # Итерируемся, пока не найдем все потенциалы
        while np.isnan(u).any() or np.isnan(v).any():
            for i in range(self.m):
                for j in range(self.n):
                    # Клетка считается базисной, если в ней есть груз (даже фиктивный 0)
                    if self.x[i, j] > 1e-12:
                        if not np.isnan(u[i]) and np.isnan(v[j]):
                            v[j] = self.costs[i, j] - u[i]
                        elif not np.isnan(v[j]) and np.isnan(u[i]):
                            u[i] = self.costs[i, j] - v[j]
        return u, v

    def get_cycle(self, start_node):
        """Поиск замкнутого цикла (DFS) для перераспределения груза"""

        def get_neighbors(node, is_vertical):
            r, c = node
            if is_vertical:
                return [(i, c) for i in range(self.m) if i != r and self.x[i, c] > 0]
            else:
                return [(r, j) for j in range(self.n) if j != c and self.x[r, j] > 0]

        stack = [(start_node, [start_node], True), (start_node, [start_node], False)]
        while stack:
            curr, path, is_vert = stack.pop()
            for next_node in get_neighbors(curr, is_vert):
                if next_node == start_node and len(path) >= 4:
                    return path
                if next_node not in path:
                    stack.append((next_node, path + [next_node], not is_vert))
        return None

    def optimize(self):
        """Фаза 2: Метод потенциалов с итерациями до оптимума"""
        while True:
            # Борьба с вырожденностью: должно быть m + n - 1 занятых клеток
            num_active = np.count_nonzero(self.x > 1e-12)
            if num_active < self.m + self.n - 1:
                for i in range(self.m):
                    for j in range(self.n):
                        if self.x[i, j] == 0:
                            self.x[i, j] = 1e-13  # "Фиктивный ноль"
                            if np.count_nonzero(self.x > 1e-12) == self.m + self.n - 1:
                                break
                    else:
                        continue
                    break

            u, v = self.calculate_potentials()
            deltas = np.zeros((self.m, self.n))
            max_delta, pivot_cell = -1.0, None

            # Ищем клетку с максимальной положительной оценкой
            for i in range(self.m):
                for j in range(self.n):
                    if self.x[i, j] <= 1e-12:
                        deltas[i, j] = u[i] + v[j] - self.costs[i, j]
                        if deltas[i, j] > max_delta:
                            max_delta, pivot_cell = deltas[i, j], (i, j)

            # Критерий оптимальности: все Delta <= 0
            if pivot_cell is None or max_delta <= 1e-7:
                self._save_snapshot("План оптимален!", u, v, deltas)
                break

            self._save_snapshot(
                f"Улучшение: клетка {pivot_cell}, Delta={max_delta:.1f}", u, v, deltas
            )

            # Поиск цикла и переброска груза
            cycle = self.get_cycle(pivot_cell)
            if not cycle:
                break  # Страховка

            minus_cells = cycle[1::2]
            theta = min(self.x[r, c] for r, c in minus_cells)

            for idx, (r, c) in enumerate(cycle):
                if idx % 2 == 0:
                    self.x[r, c] += theta
                else:
                    self.x[r, c] -= theta

            # Очистка фиктивных нулей после переброски
            self.x[self.x < 1e-11] = 0

        return self.snapshots


if __name__ == "__main__":
    # Данные Варианта 3
    costs = [[2, 4, 5, 1], [2, 3, 9, 4], [3, 4, 22, 5]]
    supply = [60, 70, 20]
    demand = [40, 30, 30, 50]

    engine = TransportationInteractive(costs, supply, demand)
    engine.run_nw_corner()
    snaps = engine.optimize()

    visualize_transport(snaps, np.array(costs))
