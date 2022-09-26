import numpy as np
import utils
import pandas as pd

np.random.seed(42)

epsilon = 0.1
delta = 0.1

init_states = pd.read_csv("init_stat_100.csv", delimiter=",", header=None)
dones = pd.read_csv("done_100.csv", delimiter=",", header=None)
init_states = init_states.sort_values(by=[0, 1, 2, 3])
dones = dones.iloc[init_states.index.values]
breakpoint()

X_range = [-0.6, -0.59]
V_range = [0, 0]
C_range = [-1, -0.4]
D_range = [-0.01, -0.005]

dones = dones[(init_states.iloc[:, 0] >= X_range[0]) & (init_states.iloc[:, 0] <= X_range[1]) &
              (init_states.iloc[:, 1] >= V_range[0]) & (init_states.iloc[:, 1] <= V_range[1]) &
              (init_states.iloc[:, 2] >= C_range[0]) & (init_states.iloc[:, 2] <= C_range[1]) &
              (init_states.iloc[:, 3] >= D_range[0]) & (init_states.iloc[:, 3] <= D_range[1])]

