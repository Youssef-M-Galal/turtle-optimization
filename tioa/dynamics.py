import numpy as np


def compute_step_size(i, neighbors, fitness, S_max, k):
    neighbors_fitness = fitness[neighbors]
    f_i = fitness[i]
    f_neighbors_mean = np.mean(neighbors_fitness)
    p_i = abs(f_i - f_neighbors_mean) / (1 + abs(f_neighbors_mean))
    S_i = S_max * np.exp(-k * p_i)
    return S_i


def compute_exploration(t, T, E0):
    E_i = E0 * (1 - t / T)
    return E_i
# --------------------------------------------------
# compute_step_size function
# Pressure-Aware Step Size (PAS)
# --------------------------------------------------
# Purpose:
# Adapt step size based on fitness pressure.
#
# Step 1: Compute pressure
# P_i = |f_i - f_neighbors_mean| / (1 + |f_neighbors_mean|)
#
# Step 2: Compute step size
# S_i = S_max * exp(-k * P_i)
#
# Where:
# f_i              = fitness of current turtle
# f_neighbors_mean = average fitness of neighbors
# S_max            = maximum step size
# k                = decay coefficient
#
# Intuition:
# - High pressure → small step (careful movement)
# - Low pressure  → large step (faster exploration)
#
# Output:
# Scalar step size S_i




# --------------------------------------------------
# compute_exploration function
# Temperature-Driven Exploration (TDE)
# --------------------------------------------------
# Purpose:
# Control exploration over time.
#
# Equation:
# E_i = E0 * (1 - t / T)
#
# Where:
# E0 = initial exploration strength
# t  = current iteration
# T  = total iterations
#
# Intuition:
# - Early → high exploration
# - Late  → low exploration (focus on exploitation)
#
# Output:
# Scalar exploration factor E_i
