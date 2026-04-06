import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def visualize_simplex(snapshots):
    fig, ax = plt.subplots(figsize=(9, 7))
    x_range = np.linspace(0, 5, 400)

    # Static constraints visualization
    ax.plot(x_range, (5 * x_range - 3) / 2, "b--", alpha=0.3, label="5x1 - 2x2 <= 3")
    ax.plot(x_range, 1 - x_range, "g--", alpha=0.3, label="x1 + x2 >= 1")
    ax.plot(x_range, 3 + 3 * x_range, "r--", alpha=0.3, label="-3x1 + x2 <= 3")
    ax.plot(x_range, 4 - 2 * x_range, "m--", alpha=0.3, label="2x1 + x2 <= 4")

    # Fill Feasible Region (ODR)
    y_min = np.maximum(np.maximum((5 * x_range - 3) / 2, 1 - x_range), 0)
    y_max = np.minimum(3 + 3 * x_range, 4 - 2 * x_range)
    ax.fill_between(
        x_range, y_min, y_max, where=(y_max >= y_min), color="gray", alpha=0.15
    )

    # Solution point and path
    path_x, path_y = [], []
    (line_path,) = ax.plot([], [], "r:", lw=1, alpha=0.6)
    (current_dot,) = ax.plot(
        [], [], "ro", markersize=12, label="Current Basis", zorder=5
    )

    state = {"step": 0}

    def update():
        step = state["step"]
        tbl = snapshots[step]

        # Extract x1 and x2 from basis
        x1, x2 = 0.0, 0.0
        for c in [0, 1]:  # Columns for x1, x2
            col = tbl[:4, c]
            if np.sum(col == 1) == 1 and np.sum(col == 0) == 3:
                val = tbl[np.where(col == 1)[0][0], -1]
                if c == 0:
                    x1 = val
                else:
                    x2 = val

        path_x.append(x1)
        path_y.append(x2)

        current_dot.set_data([x1], [x2])
        line_path.set_data(path_x[: step + 1], path_y[: step + 1])

        # Display current objective value
        f_val = tbl[4, -1] if step > 0 else 0
        ax.set_title(
            f"Simplex Iteration: {step} / {len(snapshots)-1}\n"
            f"Current Point: ({x1:.2f}, {x2:.2f}) | F = {f_val:.2f}\n"
            f"Controls: [Left/Right Arrows]"
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            state["step"] = min(state["step"] + 1, len(snapshots) - 1)
        elif event.key == "left":
            state["step"] = max(state["step"] - 1, 0)
            path_x.pop()  # Remove path history for visual undo
            path_y.pop()
        update()

    fig.canvas.mpl_connect("key_press_event", on_key)
    update()

    plt.xlim(0, 3)
    plt.ylim(0, 5)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.show()


def visualize_with_matrix(snapshots, columns, row_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    x_range = np.linspace(0, 5, 400)

    # Static constraints and Feasible Region [cite: 2026-03-20]
    ax.plot(x_range, (5 * x_range - 3) / 2, "b--", alpha=0.4, label="5x1 - 2x2 <= 3")
    ax.plot(x_range, 1 - x_range, "g--", alpha=0.4, label="x1 + x2 >= 1")
    ax.plot(x_range, 3 + 3 * x_range, "r--", alpha=0.4, label="-3x1 + x2 <= 3")
    ax.plot(x_range, 4 - 2 * x_range, "m--", alpha=0.4, label="2x1 + x2 <= 4")

    y_min = np.maximum(np.maximum((5 * x_range - 3) / 2, 1 - x_range), 0)
    y_max = np.minimum(3 + 3 * x_range, 4 - 2 * x_range)
    ax.fill_between(
        x_range, y_min, y_max, where=(y_max >= y_min), color="gray", alpha=0.1
    )

    # Gradient Vector (F = 7x1 - 2x2)
    ax.quiver(
        0.5, 3.5, 7, -2, color="red", scale=40, label="Gradient F (Direction of Max)"
    )

    (current_dot,) = ax.plot([], [], "ro", markersize=12, zorder=5)
    state = {"step": 0}

    def show_matrix(step):
        print(f"\n--- Matrix at Step {step} ---")
        df = pd.DataFrame(snapshots[step], columns=columns, index=row_names)
        print(df.round(2))

    def update():
        step = state["step"]
        tbl = snapshots[step]

        # Extract basic x1, x2 [cite: 2026-03-20]
        coords = [0.0, 0.0]
        for c in [0, 1]:
            col = tbl[:4, c]
            if np.count_nonzero(col == 1) == 1 and np.count_nonzero(col == 0) == 3:
                coords[c] = tbl[np.where(col == 1)[0][0], -1]

        current_dot.set_data([coords[0]], [coords[1]])
        ax.set_title(
            f"Iteration: {step} | Point: ({coords[0]:.2f}, {coords[1]:.2f})\n"
            f"Objective F: {tbl[4, -1]:.2f} | Press -> to Step"
        )
        show_matrix(step)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            state["step"] = min(state["step"] + 1, len(snapshots) - 1)
        elif event.key == "left":
            state["step"] = max(state["step"] - 1, 0)
        update()

    fig.canvas.mpl_connect("key_press_event", on_key)
    update()
    plt.xlim(0, 3)
    plt.ylim(0, 5)
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()
