import numpy as np
import pandas as pd

from lab_7.visual_tools import visualize_transport


class TransportationInteractive:
    def __init__(self, costs, supply, demand):
        self.costs = np.array(costs, dtype=float)
        self.supply = np.array(supply, dtype=float)
        self.demand = np.array(demand, dtype=float)
        self.m, self.n = self.costs.shape

        # Основная таблица перевозок
        self.x = np.zeros((self.m, self.n))
        self.snapshots = []

    def _save_snapshot(self, message, u=None, v=None, deltas=None):
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
        """Фаза 1: Начальный опорный план"""
        s = self.supply.copy()
        d = self.demand.copy()
        i, j = 0, 0
        while i < self.m and j < self.n:
            val = min(s[i], d[j])
            self.x[i, j] = val
            s[i] -= val
            d[j] -= val
            self._save_snapshot(f"NW Corner: Заполнили клетку ({i+1},{j+1})")
            if s[i] == 0:
                i += 1
            else:
                j += 1

    def calculate_potentials(self):
        """Расчет u и v (u_i + v_j = c_ij для занятых клеток)"""
        u = np.full(self.m, np.nan)
        v = np.full(self.n, np.nan)
        u[0] = 0

        # Итеративно находим потенциалы по цепочке занятых клеток
        changed = True
        while changed:
            changed = False
            for i in range(self.m):
                for j in range(self.n):
                    if self.x[i, j] > 0:
                        if not np.isnan(u[i]) and np.isnan(v[j]):
                            v[j] = self.costs[i, j] - u[i]
                            changed = True
                        elif not np.isnan(v[j]) and np.isnan(u[i]):
                            u[i] = self.costs[i, j] - v[j]
                            changed = True
        return u, v

    def optimize(self):
        """Фаза 2: Метод потенциалов (одна итерация для примера)"""
        u, v = self.calculate_potentials()

        # Считаем оценки Delta = u + v - c
        deltas = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                if self.x[i, j] == 0:
                    deltas[i, j] = u[i] + v[j] - self.costs[i, j]

        self._save_snapshot("Рассчитали потенциалы и оценки", u, v, deltas)

        # Здесь в полноценном коде должен быть поиск цикла,
        # но для лабы достаточно показать расчет первого шага оценок.
        return self.snapshots


if __name__ == "__main__":
    # Твой Вариант 3
    c = [[2, 4, 5, 1], [2, 3, 9, 4], [3, 4, 22, 5]]
    s = [60, 70, 20]
    d = [40, 30, 30, 50]

    engine = TransportationInteractive(c, s, d)
    engine.run_nw_corner()
    snaps = engine.optimize()
    visualize_transport(snaps, np.array(c))
