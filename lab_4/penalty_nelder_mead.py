import numpy as np
from lab_2.visual_tools import visualize_nelder_mead


class NelderMead2D:
    def __init__(
        self,
        func,
        start_point=[1.0, 1.0],
        step=0.1,
        constraint_func=None,
        penalty_weight=100.0,
    ):
        self.func = func
        # Function defining the constraint: g(x) <= 0 is allowed, g(x) > 0 is penalized
        self.constraint_func = constraint_func
        # Penalty multiplier to discourage stepping into forbidden zones
        self.penalty_weight = penalty_weight

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

    def _get_f_val(self, x):
        """Calculate function value with optional external penalty"""
        f_val = self.func(x)

        # Apply penalty if constraint is violated (g(x) > 0)
        if self.constraint_func is not None:
            g_val = self.constraint_func(x)
            if g_val > 0:
                penalty = self.penalty_weight * (g_val**2)
                # Add quadratic penalty based on the magnitude of violation
                print(f"Penalty triggered at {x}: {penalty}")  # Temporary debug
                f_val += penalty

        return f_val

    def calculate(self, max_iter=100) -> list:
        alpha_reflection, gamma_expansion, rho_shrink, sigma_reduction = (
            1.0,
            2.0,
            0.5,
            0.5,
        )

        for iter in range(max_iter):
            # Sorting based on penalized function values
            vertices_func_vals = np.array(
                [self._get_f_val(point) for point in self.simplex]
            )

            # Fancy indexing, simplex now contains point by descending f-vals
            sorted_func_vals_idxs = np.argsort(vertices_func_vals)

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

            # Reflect worst point using penalized evaluation
            reflected_p = centroid + alpha_reflection * (centroid - worst_p)
            f_reflected = self._get_f_val(reflected_p)

            # Expand if reflected is the new best
            if f_reflected < f_best:
                expanded_p = centroid + gamma_expansion * (reflected_p - centroid)
                f_expanded = self._get_f_val(expanded_p)

                self.simplex[2] = (
                    expanded_p if f_expanded < f_reflected else reflected_p
                )

            # Accept reflected if it's only better than mid
            elif f_reflected < f_mid:
                self.simplex[2] = reflected_p

            else:
                # Contract in the best direction
                point_for_contraction = (
                    reflected_p if f_reflected < f_worst else worst_p
                )

                contracted_p = centroid + rho_shrink * (
                    point_for_contraction - centroid
                )
                f_contracted = self._get_f_val(contracted_p)

                if f_contracted < f_worst:
                    self.simplex[2] = contracted_p
                else:
                    # Shrink towards best point
                    for point in range(1, len(self.simplex)):
                        self.simplex[point] = best_p + sigma_reduction * (
                            self.simplex[point] - best_p
                        )

            self.simplex_snapshots.append(self.simplex.copy())

            # Convergence by standard deviation of penalized values
            if np.std(vertices_func_vals) < 1e-6:
                print(
                    f"Function minimum: {self.func(self.simplex[0])}. Simplex points {self.simplex}. Iterations: {iter + 1}"
                )
                break

        return self.simplex_snapshots


def two_vars_function(x: np.array):
    x1, x2 = x
    return np.exp(x1**2) + (x1 + x2) ** 2


# Example constraint: point must be within a certain radius, e.g., x1^2 + x2^2 <= 4
# We return x1^2 + x2^2 - 4 so that it is > 0 when the constraint is broken
def my_constraint(x):
    return x[0] ** 2 + x[1] ** 2 - 4


# def my_constraint(x):
#     # This makes (0,0) forbidden because -(0+0) + 0.5 = 0.5 (which is > 0)
#     # The allowed zone is where x1 + x2 > 0.5
#     return 0.5 - (x[0] + x[1])


# Initialize with the constraint function
nm2d = NelderMead2D(func=two_vars_function, constraint_func=my_constraint)
snaps = nm2d.calculate(max_iter=100)
visualize_nelder_mead(two_vars_function, snaps)
