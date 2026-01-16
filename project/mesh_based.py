import time
import numpy as np

G = 1.0  # gravity constant

def mesh_based_algorithm(positions, velocities, masses, dt, grid_size=10):
    time_simul = time.time()
    n = len(masses)
    accelerations = np.zeros_like(positions)
    checks = 0
    
    # Determine grid bounds
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    cell_size = (max_pos - min_pos) / grid_size
    cell_size = np.where(cell_size > 1e-10, cell_size, 1.0)
    
    # Assign bodies to grid cells
    cell_indices = ((positions - min_pos) / cell_size).astype(int)
    cell_indices = np.clip(cell_indices, 0, grid_size - 1)
    
    # Create dictionary of cells with their bodies
    cells = {}
    for i in range(n):
        cell_key = tuple(cell_indices[i])
        if cell_key not in cells:
            cells[cell_key] = []
        cells[cell_key].append(i)
    
    # Calculate center of mass for each cell
    cell_com = {}
    cell_com_mass = {}
    for cell_key, body_indices in cells.items():
        total_mass = masses[body_indices].sum()
        com = np.average(positions[body_indices], axis=0, weights=masses[body_indices])
        cell_com[cell_key] = com
        cell_com_mass[cell_key] = total_mass
    
    # Calculate accelerations using cell centers of mass
    for i in range(n):
        cell_key = tuple(cell_indices[i])
        for other_cell_key, other_mass in cell_com_mass.items():
            if cell_key != other_cell_key:
                checks += 1
                r_vec = cell_com[other_cell_key] - positions[i]
                r_mag = np.linalg.norm(r_vec)
                if r_mag > 1e-10:
                    accelerations[i] += G * other_mass * r_vec / r_mag**3
    
    time_simul = time.time() - time_simul
    
    velocities += accelerations * dt
    positions += velocities * dt
    return positions, velocities, time_simul, checks


def mesh_based_poisson_method(positions, velocities, masses, dt, grid_size=10):
    n = len(masses)
    accelerations = np.zeros_like(positions)
    
    time_simul = time.time()
    checks = 0
    # Determine grid bounds
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    cell_size = (max_pos - min_pos) / grid_size
    cell_size = np.where(cell_size > 1e-10, cell_size, 1.0)
    
    # Assign bodies to grid cells
    cell_indices = ((positions - min_pos) / cell_size).astype(int)
    cell_indices = np.clip(cell_indices, 0, grid_size - 1)
    
    # Create density grid
    density_grid = np.zeros((grid_size, grid_size))
    for i in range(n):
        cell_key = tuple(cell_indices[i])
        density_grid[cell_key] += masses[i]
    
    # Solve Poisson equation on the grid (using FFT or finite difference method)
    # Placeholder for Poisson solver
    multipliers = np.fft.fftfreq(grid_size).reshape(-1, 1), np.fft.fftfreq(grid_size).reshape(1, -1)
    multiplier_array = -(4 * np.pi) * (multipliers[0]**2 + multipliers[1]**2) * 5000
    multiplier_array[0, 0] = 1.0  # avoid division by zero
    in_fourier_space = np.fft.fft2(density_grid) 
    #in_fourier_space[0, 0] = 0.0  # set mean to zero
    in_fourier_space /= multiplier_array
    potential_grid = np.fft.ifft2(in_fourier_space).real
    
    grad_potential = np.gradient(potential_grid)
    # Calculate accelerations from potential
    for i in range(n):
        checks += 1
        cell_key = tuple(cell_indices[i])
        grad_potential_at_cell = np.array([grad[cell_key] for grad in grad_potential])
        accelerations[i] = -grad_potential_at_cell
    
    time_simul = time.time() - time_simul
    velocities += accelerations * dt
    positions += velocities * dt
    return positions, velocities, time_simul, checks