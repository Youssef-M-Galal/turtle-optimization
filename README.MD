# 🐢 Turtle Inspired Optimization Algorithm (TIOA) – Full Project Guide

---

# 🎯 1. PURPOSE OF THIS DOCUMENT

This document explains:

* The full flow of the TIOA project
* The role of each module
* Mathematical meaning of each step
* Data shapes and consistency rules
* Integration rules between teammates

This ensures:
✔ No conflicts
✔ Clear understanding
✔ Consistent implementation

---

# 🧠 2. HIGH-LEVEL IDEA

TIOA is a **population-based metaheuristic algorithm**.

We simulate a group of turtles exploring a search space to find the **optimal solution**.

Each turtle:

* has a position (solution)
* evaluates its quality (fitness)
* moves based on:

  * neighbors (local)
  * randomness (exploration)
  * best solutions (global guidance)

---

# 🧱 3. DATA STRUCTURE (CRITICAL)

## Global variables:

positions → shape (N, dim)
fitness → shape (N,)
pbest_positions → shape (N, dim)
pbest_fitness → shape (N,)

---

## Meaning:

* N = number of turtles
* dim = dimension of problem

---

## Example:

If:
N = 5
dim = 2

positions =
[
[x1, y1],
[x2, y2],
...
]

---

## Individual turtle:

positions[i] → shape (dim,)
fitness[i] → scalar

---

# ⚠️ DIMENSION CONSISTENCY RULES

These MUST NEVER break:

1. positions[i] → (dim,)
2. direction D_i → (dim,)
3. random vector R_i → (dim,)
4. migration M_i → (dim,)
5. delta_X → (dim,)

✔ All vector operations must match dimension

---

# 🧠 4. PROJECT STRUCTURE

core/
optimizer.py   → main algorithm loop
population.py  → initialization
fitness.py     → objective function

tioa/
neighborhood.py → neighbors
dynamics.py     → PAS + TDE
movement.py     → direction + migration + update

utils/
helpers.py      → utilities

main.py → entry point
config.py → parameters

---

# 🔄 5. FULL ALGORITHM FLOW

---

## STEP 1: INITIALIZATION

positions = random values in bounds
fitness = evaluate objective function

pbest_positions = positions
pbest_fitness = fitness

X_best = best turtle
f_best = best fitness

---

## STEP 2: MAIN LOOP

for t in range(T):

```
for each turtle i:
```

---

## STEP 3: NEIGHBORHOOD

Find nearby turtles:

neighbors = { j | distance(X_i, X_j) < r }

Where:
r = alpha * max_distance

---

## STEP 4: DIRECTION

Move toward neighbors:

D_i = mean(X_neighbors) - X_i

---

## STEP 5: STEP SIZE (PAS)

Compute pressure:

P_i = |f_i - mean(f_neighbors)| / (1 + |mean(f_neighbors)|)

Step size:

S_i = S_max * exp(-k * P_i)

---

## STEP 6: EXPLORATION (TDE)

E_i = E0 * (1 - t / T)

---

## STEP 7: RANDOM VECTOR

R_i ~ Uniform(-1, 1)

---

## STEP 8: MIGRATION (MH)

Only applied periodically:

if t % T_m == 0:

M_i = μ (X_best - X_i) + ν (X_pbest - X_i)

else:

M_i = 0

---

## STEP 9: FINAL UPDATE

ΔX_i = S_i * D_i + E_i * R_i + M_i

X_i = X_i + ΔX_i

---

## STEP 10: BOUNDS

Clip:

X_i = clip(X_i, lower, upper)

---

## STEP 11: FITNESS

fitness[i] = f(X_i)

---

## STEP 12: PERSONAL BEST

if fitness[i] < pbest_fitness[i]:

```
update pbest
```

---

## STEP 13: GLOBAL BEST

Find best turtle in population

---

# 🧠 6. INTUITION BEHIND EACH COMPONENT

---

## Direction → Local Learning

Move toward neighbors

---

## PAS → Adaptive Step

Large difference → cautious movement
Small difference → aggressive movement

---

## TDE → Exploration Control

Early → random exploration
Late → precise exploitation

---

## Migration → Global Guidance

Push toward:

* best solution
* personal memory

---

# 🧠 7. COMPLETE UPDATE EQUATION

X_i = X_i + S_i D_i + E_i R_i + M_i

---

# 🧠 8. FUNCTION CONTRACTS (IMPORTANT)

Each function must respect:

---

compute_direction
Input: (i, neighbors, positions)
Output: (dim,)

---

compute_step_size
Input: (i, neighbors, fitness, S_max, k)
Output: scalar

---

compute_exploration
Input: (t, T, E0)
Output: scalar

---

compute_migration
Input: (X_i, X_best, X_pbest, mu, nu)
Output: (dim,)

---

compute_update
Input: vectors + scalars
Output: (dim,)

---

# ⚠️ 9. COMMON ERRORS TO AVOID

❌ Shape mismatch
❌ Returning scalar instead of vector
❌ Modifying original arrays unintentionally
❌ Empty neighbors without handling
❌ Forgetting bounds

---
