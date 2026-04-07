import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Добавляем функцию для красивого вывода текущего состояния переменных
def get_current_variables(table, cols):
    """
    Extracts the values of all variables from the simplex table.
    If a column is a unit vector, the variable is basic.
    Otherwise, it is non-basic (equal to zero).
    """
    var_values = {col: 0.0 for col in cols[:-1]}  # Initially all are 0 (non-basic)

    for c_idx, col_name in enumerate(cols[:-1]):
        col_data = table[:4, c_idx]  # Check only constraint rows
        # A variable is basic if its column is a unit vector (one '1', rest '0')
        if np.sum(col_data == 1) == 1 and np.sum(col_data == 0) == 3:
            row_idx = np.where(col_data == 1)[0][0]
            var_values[col_name] = table[row_idx, -1]  # Take value from 'Res' column

    return var_values


def visualize_with_matrix(snapshots, columns, row_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    x_range = np.linspace(0, 5, 400)

    # Static plot elements
    ax.plot(x_range, (5 * x_range - 3) / 2, "b--", alpha=0.3, label="5x1 - 2x2 <= 3")
    ax.plot(x_range, 1 - x_range, "g--", alpha=0.3, label="x1 + x2 >= 1")
    ax.plot(x_range, 3 + 3 * x_range, "r--", alpha=0.3, label="-3x1 + x2 <= 3")
    ax.plot(x_range, 4 - 2 * x_range, "m--", alpha=0.3, label="2x1 + x2 <= 4")

    y_min = np.maximum(np.maximum((5 * x_range - 3) / 2, 1 - x_range), 0)
    y_max = np.minimum(3 + 3 * x_range, 4 - 2 * x_range)
    ax.fill_between(
        x_range, y_min, y_max, where=(y_max >= y_min), color="gray", alpha=0.1
    )

    # Objective Gradient
    ax.quiver(0.5, 4, 7, -2, color="red", scale=40, label="Gradient F")

    (current_dot,) = ax.plot([], [], "ro", markersize=12, zorder=5)
    state = {"step": 0}

    def update():
        step = state["step"]
        tbl = snapshots[step]

        # Get variable states
        var_states = get_current_variables(tbl, columns)

        # Update plot point
        current_dot.set_data([var_states["x1"]], [var_states["x2"]])

        # Console output for the defense
        print(f"\n{'='*40}")
        print(f"STEP {step}: Current Vertex Analysis")
        print(f"{'='*40}")

        # Print Simplex Table
        df = pd.DataFrame(tbl, columns=columns, index=row_names)
        print("\nSimplex Tableau:")
        print(df.round(2))

        print("\nVariable Status:")
        basic = []
        non_basic = []
        for var, val in var_states.items():
            if val != 0:
                basic.append(f"{var} = {val:.2f}")
            else:
                non_basic.append(var)

        print(f"  [ACTIVE] Basic variables: {', '.join(basic)}")
        print(f"  [OFF] Non-basic (Zeros): {', '.join(non_basic)}")
        print(f"  Current Objective F: {tbl[4, -1]:.2f}")

        ax.set_title(
            f"Step {step} | x1={var_states['x1']:.2f}, x2={var_states['x2']:.2f} | F={tbl[4, -1]:.2f}"
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
    plt.xlim(0, 3)
    plt.ylim(0, 5)
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()
