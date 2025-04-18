# teps_animation.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines
from collections import deque

# Phase color mapping
# Phase color mapping
phase_colors = {
    0: "#3b3b3b",   #dark grey
    1: "#00A67C",  # black
    2: "#0076CC",  # red
    3: "#E51C44",  # green
    4: "#F9B32F",  # blue
    5: "#000000"  # yellow
}

# Legend for phases
phase_labels = {
    0: "Rapid",
    1: "Air-Drilling",
    2: "Drilling",
    3: "Unexpected Drop",
    4: "Expected Drop",
    5: "Reposition"
    }

def run_teps_animation(get_next_sample, window_size=1000, update_interval_ms=10):
    xdata = list(range(window_size))
    ydata = [0] * window_size
    phase_history = deque([0] * window_size, maxlen=window_size)

    # Setup plot
    fig, ax = plt.subplots()
    ax.set_ylim(-1, 5)
    ax.set_xlim(0, window_size)
    ax.set_title("TEPS Demo (Streaming at 100Hz)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Torque")

    # Line setup
    line_collection = LineCollection([], colors=["#000000"] * (window_size - 1), linewidths=2)
    ax.add_collection(line_collection)

    # Legend
    legend_lines = [
        mlines.Line2D([], [], color=phase_colors[phase], linewidth=4,
                      label=f"{phase_labels[phase]}")
        for phase in sorted(phase_colors)
    ]
    ax.legend(handles=legend_lines,loc="upper left")

    def update(frame):
        sample = get_next_sample()
        if sample is None:
            ani.event_source.stop()
            return line_collection,

        value, phase = sample
        ydata.append(value)
        ydata.pop(0)
        phase_history.append(phase)

        points = np.array([xdata, ydata]).T
        segments = np.array([[points[i], points[i+1]] for i in range(len(points)-1)])
        colors = [phase_colors.get(p, "#000000") for p in list(phase_history)[1:]]

        line_collection.set_segments(segments)
        line_collection.set_color(colors)

        return line_collection,

    ani = animation.FuncAnimation(
        fig,
        update,
        interval=update_interval_ms,
        blit=True,
        cache_frame_data=False
    )
    #ani.save("teps_demo.gif", writer="pillow", fps=  1000 // update_interval_ms)
    plt.show()


