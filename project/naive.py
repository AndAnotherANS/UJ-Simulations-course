import time
import numpy as np

G = 1.0 # gravity constant

def naive_algorithm(positions, velocities, masses, dt):
    time_simul = time.time()
    n = len(masses)
    accelerations = np.zeros_like(positions)
    checks = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                checks += 1
                r_vec = positions[j] - positions[i]
                r_mag = np.linalg.norm(r_vec)
                if r_mag > 1e-10:
                    accelerations[i] += G * masses[j] * r_vec / r_mag**3
    time_simul = time.time() - time_simul

    velocities += accelerations * dt
    positions += velocities * dt
    return positions, velocities, time_simul, checks

