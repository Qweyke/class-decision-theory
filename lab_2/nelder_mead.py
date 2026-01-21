from matplotlib import pyplot as plt
import numpy as np


def two_vars_function(x: np.array):
    x1, x2 = x

    return np.exp(x1**2) + (x1 + x2) ** 2


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
        self.simplex_snapshots = [self.simplex.copy()]

    def calculate(self, max_iter=100) -> list:
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        for iter in range(max_iter):
            # Sorting
            func_vals = np.array([self.func(point) for point in self.simplex])
            ordered_p_ids = np.argsort(func_vals)

            # Fancy indexing, simplex now contains point by descending f-vals
            self.simplex = self.simplex[ordered_p_ids]
            func_vals = func_vals[ordered_p_ids]

            best_p, mid_p, worst_p = self.simplex[0], self.simplex[1], self.simplex[2]
            f_best, f_mid, f_worst = func_vals[0], func_vals[1], func_vals[2]

            centroid = (best_p + mid_p) / 2.0

            # Reflect worst point in centroid's direction
            reflected_p = centroid + alpha * (centroid - worst_p)
            f_reflected = self.func(reflected_p)

            # Expand by gamma length in reflected direction, if extended is the new best
            if f_reflected < f_best:
                expanded_p = centroid + gamma * (reflected_p - centroid)
                f_expanded = self.func(expanded_p)

                # Keep reflected, if expansion is lost-cause
                self.simplex[2] = (
                    expanded_p if f_expanded < f_reflected else reflected_p
                )

            elif f_reflected < f_mid:
                self.simplex[2] = reflected_p

            else:
                # Contract in the best direction, if reflected point is worse than mid and best
                contracted_p = centroid + rho * (
                    (reflected_p if f_reflected < f_worst else worst_p) - centroid
                )
                f_contracted = self.func(contracted_p)

                if f_contracted < f_worst:
                    self.simplex[2] = contracted_p

                # Shrink if contracted point is the new worst
                else:
                    for j in range(1, len(self.simplex)):
                        self.simplex[j] = best_p + sigma * (self.simplex[j] - best_p)

            self.simplex_snapshots.append(self.simplex.copy())

            # Convergence by standard deviation
            if np.std(func_vals) < 1e-6:
                print(
                    f"Function minimum: {self.func(self.simplex[0])}. Simplex points {self.simplex}. Iterations: {iter + 1}"
                )
                break

        return self.simplex_snapshots


def visualize_nelder_mead(func, snapshots):
    fig, ax = plt.subplots(figsize=(8, 8))

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func(np.array([xi, yi])) for xi in x] for yi in y])
    ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.4)

    # Main simplex
    (line,) = ax.plot([], [], "r-o", lw=2, markersize=5, zorder=10)
    ghosts = {}

    state = {"step": 0}

    def update():
        step = state["step"]

        pts = snapshots[step]
        closed_pts = np.vstack([pts, pts[0]])
        line.set_data(closed_pts[:, 0], closed_pts[:, 1])

        for g in ghosts.values():
            g.set_visible(False)

        for i in range(step):
            if i not in ghosts:
                prev_pts = snapshots[i]
                c_prev = np.vstack([prev_pts, prev_pts[0]])
                (g_line,) = ax.plot(
                    c_prev[:, 0], c_prev[:, 1], "k--", lw=1, alpha=0.1, zorder=1
                )
                ghosts[i] = g_line
            else:
                ghosts[i].set_visible(True)

        new_title = (
            f"Iter: {step} / {len(snapshots)-1}\n" f"Controls: <- and -> to navigate"
        )
        ax.set_title(new_title, fontsize=12, pad=10)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            state["step"] = min(state["step"] + 1, len(snapshots) - 1)
        elif event.key == "left":
            state["step"] = max(state["step"] - 1, 0)
        update()

    fig.canvas.mpl_connect("key_press_event", on_key)
    update()
    plt.show()


nm2d = NelderMead2D(func=two_vars_function)
snaps = nm2d.calculate(max_iter=100)
visualize_nelder_mead(two_vars_function, snaps)
