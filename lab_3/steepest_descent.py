from matplotlib import pyplot as plt
import numpy as np
import sympy as sp

from visual_tools import visualize_gradient_descent


class GradientDescent2D:
    def __init__(self, func, numerical_derivative=False, start_point=[1.0, 1.0]):
        self.func = func
        self.start_point = np.array(start_point)
        self.calculate_grad = (
            self._get_numerical_gradient
            if numerical_derivative
            else self._get_sympy_gradient
        )

    def _get_sympy_gradient(self, point: np.array):
        x1_sym, x2_sym = sp.Symbol("x1"), sp.Symbol("x2")
        f_sym = self.func([x1_sym, x2_sym])

        grad_sym = [sp.diff(f_sym, x1_sym), sp.diff(f_sym, x2_sym)]

        grad_func = sp.lambdify([x1_sym, x2_sym], grad_sym, "numpy")

        return np.array(grad_func(point[0], point[1]))

    def _get_numerical_gradient(self, point: np.array):
        h = 1e-7
        grad = np.zeros_like(point)

        for i in range(len(point)):
            point_forward = np.copy(point).astype(float)
            point_backward = np.copy(point).astype(float)

            point_forward[i] += h
            point_backward[i] -= h

            grad[i] = (self.func(point_forward) - self.func(point_backward)) / (2 * h)

        return grad

    def static_step_descent(self, grad_step=0.1, eps=1e-6, max_iters=1000):
        curr_point = self.start_point
        snapshots = []
        snapshots.append(curr_point.copy())

        for i in range(max_iters):
            grad = self.calculate_grad(curr_point)
            if np.linalg.norm(grad) < eps:
                break

            curr_point = curr_point - grad_step * grad
            snapshots.append(curr_point.copy())

        print(f"Func min: {self.func(curr_point)}, iters: {i + 1}")
        return snapshots

    def backtracking_line_search(self, grad_step=1.0, eps=1e-6, max_iters=1000):
        curr_point = self.start_point
        snapshots = []
        snapshots.append(curr_point.copy())

        for i in range(max_iters):
            grad = self.calculate_grad(curr_point)
            if np.linalg.norm(grad) < eps:
                break

            adjusted_step = grad_step
            while self.func(curr_point - (adjusted_step * grad)) > self.func(
                curr_point
            ):
                adjusted_step *= 0.5

                if adjusted_step < 1e-12:
                    print("Step is too small, breaking")
                    break

            curr_point = curr_point - (adjusted_step * grad)
            snapshots.append(curr_point.copy())

        print(f"Func min: {self.func(curr_point)}, iters: {i + 1}")
        return snapshots

    def steepest_descent(self, grad_step_max=1, eps=1e-6, max_iters=1000):
        def find_local_minimum(func):
            left, right = 0, grad_step_max
            gr = 0.618

            x1, x2 = left + gr * (right - left), right - gr * (right - left)
            y1, y2 = func(x1), func(x2)

            while abs(right - left) > 1e-7:
                if y1 < y2:
                    # Rearrange
                    left = x2
                    x2 = x1
                    y2 = y1

                    # Calculate new x1
                    x1 = left + gr * (right - left)
                    y1 = func(x1)

                else:
                    # Rearrange
                    right = x1
                    x1 = x2
                    y1 = y2

                    # Calculate new x2
                    x2 = right - gr * (right - left)
                    y2 = func(x2)

            return (x1 + x2) / 2

        curr_point = self.start_point
        snapshots = []
        snapshots.append(curr_point.copy())

        for i in range(max_iters):
            grad = self.calculate_grad(curr_point)
            if np.linalg.norm(grad) < eps:
                break

            calculated_step = find_local_minimum(
                func=lambda step: self.func(curr_point - (step * grad))
            )
            curr_point = curr_point - (calculated_step * grad)
            snapshots.append(curr_point.copy())

        print(f"Func min: {self.func(curr_point)}, iters: {i + 1}")

        return snapshots


def two_vars_function(x):
    x1, x2 = x

    lib = sp if isinstance(x1, sp.Symbol) else np
    return lib.exp(x1**2) + (x1 + x2) ** 2


grad_solver = GradientDescent2D(func=two_vars_function, numerical_derivative=False)

snaps_static_sd = grad_solver.static_step_descent()
snaps_backtrack_ls = grad_solver.backtracking_line_search()
snaps_steepest_d = grad_solver.steepest_descent(grad_step_max=2)

visualize_gradient_descent(
    func=two_vars_function,
    snapshots=snaps_steepest_d,
    grad_func=grad_solver.calculate_grad,
)
