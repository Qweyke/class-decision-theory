from matplotlib import pyplot as plt
import numpy as np


def visualize_gradient_descent(func, snapshots, grad_func=None):
    fig, ax = plt.subplots(figsize=(9, 8))

    all_pts = np.array(snapshots)
    x_min, x_max = all_pts[:, 0].min() - 1, all_pts[:, 0].max() + 1
    y_min, y_max = all_pts[:, 1].min() - 1, all_pts[:, 1].max() + 1

    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func(np.array([xi, yi])) for xi in x] for yi in y])

    cont = ax.contourf(X, Y, Z, levels=40, cmap="viridis", alpha=0.3)
    ax.contour(X, Y, Z, levels=40, colors="black", alpha=0.1, linewidths=0.5)
    plt.colorbar(cont)

    (path_line,) = ax.plot([], [], "r-o", lw=1.5, markersize=4, label="Path", zorder=5)

    (current_dot,) = ax.plot(
        [], [], "ro", markersize=8, markeredgecolor="white", zorder=10
    )

    quiver = None
    state = {"step": 0}

    def update():
        step = state["step"]
        pts = np.array(snapshots[: step + 1])

        path_line.set_data(pts[:, 0], pts[:, 1])
        current_dot.set_data([pts[-1, 0]], [pts[-1, 1]])

        nonlocal quiver
        if quiver:
            quiver.remove()
            quiver = None

        if step < len(snapshots) - 1 and grad_func is not None:
            p = snapshots[step]
            g = grad_func(p)
            g_norm = g / (np.linalg.norm(g) + 1e-9) * 0.3
            quiver = ax.quiver(
                p[0],
                p[1],
                -g_norm[0],
                -g_norm[1],
                color="red",
                scale=5,
                zorder=11,
                width=0.005,
            )

        ax.set_title(
            f"Gradient Descent\nIteration: {step} | Point: [{pts[-1,0]:.3f}, {pts[-1,1]:.3f}]",
            fontsize=12,
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            state["step"] = min(state["step"] + 1, len(snapshots) - 1)
        elif event.key == "left":
            state["step"] = max(state["step"] - 1, 0)
        update()

    fig.canvas.mpl_connect("key_press_event", on_key)
    update()
    plt.legend()
    plt.show()
