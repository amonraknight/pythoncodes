import numpy as np


def get_2_degree_data():
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X ** 2 + X + np.random.rand(m, 1)

    #for the drawing of the model
    mesh_size = 0.1
    x_min, x_max = X.min(), X.max()
    x_range = np.arange(x_min, x_max, mesh_size)
    x_range = x_range.reshape(-1, 1)

    return X, y, x_range
