import numpy as np
def initialize_population(N, dim, bounds):
     lower, upper = bounds
     #  (N,dim) -> N: population size, dim: number of variables in each sample
     positions = np.random.uniform(lower, upper, size=(N, dim))
     return positions

def initialize_pbest(positions, fitness):
    pbest_positions = positions.copy()
    pbest_fitness = fitness.copy()
    return pbest_positions, pbest_fitness