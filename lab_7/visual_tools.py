import matplotlib.pyplot as plt
import numpy as np


def visualize_transport(snapshots, costs):
    fig, ax = plt.subplots(figsize=(12, 7))
    state = {"step": 0}

    def update():
        step = state["step"]
        snap = snapshots[step]
        ax.clear()

        m, n = costs.shape
        table_data = []

        for i in range(m):
            row = []
            for j in range(n):
                val = snap["x"][i, j]
                cost = costs[i, j]
                # Формируем текст ячейки: груз и стоимость в скобках
                cell_text = f"{val:.0f}\n({cost:.0f})"

                # Если это фаза оптимизации и есть оценки Delta
                if "deltas" in snap and snap["x"][i, j] == 0:
                    d = snap["deltas"][i, j]
                    if not np.isnan(d) and d != 0:
                        cell_text += f"\nΔ={d:.1f}"
                row.append(cell_text)
            table_data.append(row)

        # Заголовки с потенциалами
        col_labels = [
            f"B{j+1}\nv={snap['v'][j]:.1f}" if not np.isnan(snap["v"][j]) else f"B{j+1}"
            for j in range(n)
        ]
        row_labels = [
            f"A{i+1}\nu={snap['u'][i]:.1f}" if not np.isnan(snap["u"][i]) else f"A{i+1}"
            for i in range(m)
        ]

        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            rowLabels=row_labels,
            loc="center",
            cellLoc="center",
        )

        table.scale(1, 3.5)
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        ax.axis("off")
        ax.set_title(
            f"STEP {step}: {snap['msg']}\nTotal Cost: {snap['total_cost']:.2f}", pad=20
        )

        print(f"\n[STEP {step}] {snap['msg']}")
        print(f"Cost: {snap['total_cost']}")
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
