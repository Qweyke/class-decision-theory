from matplotlib import pyplot as plt
import numpy as np


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
