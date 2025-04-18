# teps_setup.py
import numpy as np
from algo.teps import TEPS
from collections import deque
from utils.animation import run_teps_animation

# Load data
data_ = np.genfromtxt("utils/data.csv", delimiter=",", skip_header=1)
data = data_[:, 0]

# Initialize TEPS
teps = TEPS(
    init_mode='min_distance',
    min_distance_threshold=0.15,
    hist_size=20,
    factor=1
)

# TEPS state
index = 0
last_phase = 0

def get_next_sample():
    global index, last_phase

    if index >= len(data):
        return None

    value = data[index]
    TE = teps.process_sample(value)[0]
    phase = teps.get_phase_label(last_phase, TE, data_[index, 1])

    last_phase = phase
    index += 1

    return value, phase

# Run the animation
run_teps_animation(get_next_sample)
