import numpy as np
from core.population import initialize_pbest,initialize_population
from tioa.neighborhood import get_neighbors
from tioa.movement import compute_direction, compute_migration, compute_update
from tioa.dynamics import compute_step_size, compute_exploration
from utils.helpers import random_vector, clip_to_bounds

def optimize(N, dim, bounds, T, objective_function, parameters):
    """
        N: population size
        dim: sample dimension
        bounds: (l,u) the function domain
        T: number of iterations
        objective_function: function to minimize
        parameters: hyperparameters of TIOA
    """
    
    positions = initialize_population(N, dim, bounds)

    S_max = parameters.get("S_max", 0.2)
    E0 = parameters.get("E0", 0.2)
    mu = parameters.get("mu", 0.3)
    nu = parameters.get("nu", 0.2)
    alpha = parameters.get("alpha", 0.15)
    k = parameters.get("k", 2.0)
    k_min = parameters.get("k_min", 3)
    T_m = parameters.get("T_m", 10)
    
    fitness = np.zeros(N)
    
    for i in range(N):
        fitness[i] = objective_function(positions[i])
    
    pbest_positions, pbest_fitness = initialize_pbest(positions, fitness)

    best_idx = np.argmin(fitness)
    X_best = positions[best_idx].copy()
    f_best = fitness[best_idx]

    for t in range(T):
        for i in range(N):

            x_i = positions[i]

            # --------------------------------------------------
            # 1) Neighborhood (Local Interaction)
            # --------------------------------------------------
            # Identify neighboring turtles based on spatial proximity
            # neighbors = {j | distance(Xi, Xj) < threshold}
            # These neighbors influence direction and step size    
            neighbors = get_neighbors(i, positions, alpha, k_min)

             # --------------------------------------------------
            # 2) Direction Vector (Social Influence)
            # --------------------------------------------------
            # Move toward the average position of neighbors
            # Equation:
            # D_i = mean(X_neighbors) - X_i
            # This encourages local convergence
            D_i = compute_direction(i, neighbors, positions)

            # --------------------------------------------------
            # 3) Step Size (PAS - Pressure Aware Step)
            # --------------------------------------------------
            # Step size depends on fitness difference (pressure)
            # Equation:
            # P_i = |f_i - f_neighbors| / (1 + |f_neighbors|)
            # S_i = S_max * exp(-k * P_i)
            # Large pressure → small step (careful movement)
            S_i = compute_step_size(i, neighbors, fitness, S_max, k)

            # --------------------------------------------------
            # 4) Exploration (TDE - Temperature Driven)
            # --------------------------------------------------
            # Controls randomness (high early, low later)
            # Equation:
            # E_i = E0 * (1 - t / T)
            # Encourages exploration in early iterations
            E_i = compute_exploration(t, T, E0)

            # --------------------------------------------------
            # 5) Random Exploration Vector
            # --------------------------------------------------
            # Random movement component
            # Equation:
            # R_i ~ Uniform(-1, 1)
            R_i = random_vector(dim)

            # --------------------------------------------------
            # 6) Migration (MH - Global + Personal Influence)
            # --------------------------------------------------
            # Pull toward best-known solutions
            # Equation:
            # M_i = μ (X_best - X_i) + ν (X_pbest - X_i)
            # Applied periodically
            if t > 0 and t % T_m == 0:
                M_i = compute_migration(x_i, X_best, pbest_positions,i, mu, nu)
            else:
                M_i = np.zeros(dim)

            # --------------------------------------------------
            # 7) Position Update (Main Movement Equation)
            # --------------------------------------------------
            # Combine all components
            # Equation:
            # X_i = X_i + S_i * D_i + E_i * R_i + M_i
            delta_X = compute_update(x_i, S_i, D_i, E_i, R_i, M_i)

            x_i = x_i + delta_X

            # --------------------------------------------------
            # 8) Boundary Handling
            # --------------------------------------------------
            # Ensure solution stays within bounds
            x_i = clip_to_bounds(x_i, bounds)
            
            # update positions
            positions[i] = x_i
            
            # --------------------------------------------------
            # 9) Fitness Evaluation
            # --------------------------------------------------
            # Evaluate new solution
            fitness[i] = objective_function(x_i)

            # --------------------------------------------------
            # 10) Personal Best Update
            # --------------------------------------------------
            # If current position is better → update memory
            if fitness[i] < pbest_fitness[i]:
                pbest_positions[i] = positions[i].copy()
                pbest_fitness[i] = fitness[i]

            # --------------------------------------------------


        # --------------------------------------------------
        # 11) Global Best Update
        # --------------------------------------------------
        # Select best turtle in population
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < f_best:
            X_best = positions[best_idx].copy()
            f_best = fitness[best_idx]
    
    return X_best, f_best
