import numpy as np

def random_vector(dim: int) -> np.ndarray:
    return np.random.uniform(-1, 1, size=dim)

def clip_to_bounds(x, bounds):
    lower, upper = bounds
    return np.clip(x, lower, upper)

def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    return float(np.linalg.norm(x1 - x2))