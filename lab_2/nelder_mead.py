import numpy as np
from visual_tools import visualize_nelder_mead


class NelderMead2D:
    def __init__(self, func, start_point=[1.0, 1.0], step=0.1):
        self.func = func

        # Initial simplex
        self.simplex = np.array(
            [
                np.array(start_point),
                start_point + np.array([step, 0]),
                start_point + np.array([0, step]),
            ]
        )

        # Storage for simplex snapshots on every step
        self.simplex_snapshots = [self.simplex.copy()]

    def calculate(self, max_iter=100) -> list:
        alpha_reflection, gamma_expansion, rho_shrink, sigma_reduction = (
            1.0,
            2.0,
            0.5,
            0.5,
        )

        for iter in range(max_iter):
            # Sorting
            vertices_func_vals = np.array([self.func(point) for point in self.simplex])

            # Fancy indexing, simplex now contains point by descending f-vals
            sorted_func_vals_idxs = np.argsort(
                vertices_func_vals
            )  # Sort indexes by the values

            # Apply sorted indexing
            self.simplex = self.simplex[sorted_func_vals_idxs]
            vertices_func_vals = vertices_func_vals[sorted_func_vals_idxs]

            best_p, mid_p, worst_p = self.simplex[0], self.simplex[1], self.simplex[2]
            f_best, f_mid, f_worst = (
                vertices_func_vals[0],
                vertices_func_vals[1],
                vertices_func_vals[2],
            )

            # Calculate centroid
            centroid = (best_p + mid_p) / 2.0

            # Reflect worst point in centroid's direction
            reflected_p = centroid + alpha_reflection * (centroid - worst_p)
            f_reflected = self.func(reflected_p)

            # Expand by gamma_expansion length in reflected direction, if extended is the new best
            if f_reflected < f_best:
                expanded_p = centroid + gamma_expansion * (reflected_p - centroid)
                f_expanded = self.func(expanded_p)

                # Accept expanded if it's better than reflected
                self.simplex[2] = (
                    expanded_p if f_expanded < f_reflected else reflected_p
                )

            # Accept reflected if it's only better than mid
            elif f_reflected < f_mid:
                self.simplex[2] = reflected_p

            else:
                # Contract in the best direction, if reflected point is worse than mid and best
                point_for_contraction = (
                    reflected_p if f_reflected < f_worst else worst_p
                )

                contracted_p = centroid + rho_shrink * (
                    point_for_contraction - centroid
                )
                f_contracted = self.func(contracted_p)

                if f_contracted < f_worst:
                    self.simplex[2] = contracted_p

                # Shrink if contracted point is the new worst
                else:
                    for point in range(1, len(self.simplex)):
                        self.simplex[point] = best_p + sigma_reduction * (
                            self.simplex[point] - best_p
                        )

            self.simplex_snapshots.append(self.simplex.copy())

            # Convergence by standard deviation
            if np.std(vertices_func_vals, ddof=0) < 1e-6:
                print(
                    f"Function minimum: {self.func(self.simplex[0])}. Simplex points {self.simplex}. Iterations: {iter + 1}"
                )
                break

        return self.simplex_snapshots


def two_vars_function(x: np.array):
    x1, x2 = x

    return np.exp(x1**2) + (x1 + x2) ** 2


nm2d = NelderMead2D(func=two_vars_function)
snaps = nm2d.calculate(max_iter=100)
visualize_nelder_mead(two_vars_function, snaps)
