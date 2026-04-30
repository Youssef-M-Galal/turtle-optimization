import numpy as np

def objective_function(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))