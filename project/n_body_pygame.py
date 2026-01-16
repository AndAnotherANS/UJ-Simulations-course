import numpy as np
import pygame
import sys
import argparse
from pathlib import Path
import tomllib as toml
from naive import naive_algorithm as naive
from barnes_hut import calculate_acceleration as barnes_hut
from mesh_based import mesh_based_algorithm as mesh_based, mesh_based_poisson_method

pygame.init()
substeps = 1  # number of physics substeps per frame

new_mass = 9.0

def world_to_screen(pos, screen_width, screen_height, scale, center):
    x = int(screen_width / 2 + (pos[0] - center[0]) * scale)
    y = int(screen_height / 2 - (pos[1] - center[1]) * scale)
    return x, y

def simulate_pygame(positions, velocities, masses, dt, config, title="N-Body Simulation", 
                     trail_length=500, scale=1, width=800, height=800):
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()

    
    n_bodies = len(masses)
    pos = positions.copy()
    vel = velocities.copy()

    radii_game = [min(10, int(np.sqrt(masses[i]) * 2)) for i in range(n_bodies)]
    radii_world = [r / scale for r in radii_game]
    
    trails = [[] for _ in range(n_bodies)]
    lines_to_draw = []
    centers = []
    
    running = True
    paused = False
    step = 0

    new_pos = None 
    time_simul = 0
    time_collisions = 0
    time_quadtree = 0
    count = 0
    

    small_font = pygame.font.SysFont("Arial", 16)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if new_pos is None:
                    new_pos = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if new_pos is not None:
                    new_vel = (new_pos[i] - pygame.mouse.get_pos()[i] for i in (0, 1))
                    new_vel = np.array(list(new_vel))
                    world_new_pos = np.array([
                        (new_pos[0] - width / 2) / scale,
                        (height / 2 - new_pos[1]) / scale
                    ])
                    world_new_vel = np.array([
                        new_vel[0] / scale,
                        -new_vel[1] / scale
                    ])
                    pos = np.vstack([pos if pos.size else np.empty((0, 2)), world_new_pos])
                    vel = np.vstack([vel if vel.size else np.empty((0, 2)), world_new_vel])
                    masses = np.hstack([masses if masses.size else np.empty(0), new_mass])
                    radii_game = radii_game + [min(10, int(np.sqrt(new_mass) * 2))]
                    radii_world = radii_world + [radii_game[-1] / scale]
                    trails.append([])
                    new_pos = None
        if not paused:
            # update physics
            if len(masses) > 0:
                for i in range(substeps):
                    if config["algorithm"] == "naive":
                        pos, vel, time_simul, count = naive(pos, vel, masses, dt/substeps)
                    elif config["algorithm"] == "barnes_hut":
                        pos, vel, lines_to_draw, time_quadtree, time_simul, count = barnes_hut(pos, vel, masses, dt/substeps, theta=config.get("theta", 0.5))
                    elif config["algorithm"] == "mesh_based":
                        pos, vel, time_simul, count = mesh_based(pos, vel, masses, dt/substeps, grid_size=config.get("grid_size", 10))
                    elif config["algorithm"] == "mesh_based_poisson":
                        pos, vel, time_simul, count = mesh_based_poisson_method(pos, vel, masses, dt/substeps)
            else:
                lines_to_draw = []
            

            # update trails
            for i in range(len(masses)):
                trails[i].append(pos[i].copy())
                if len(trails[i]) > trail_length:
                    trails[i].pop(0)
            
            step += 1


        center = np.zeros(2)#np.average(pos, axis=0, weights=masses)
        
        screen.fill((0, 0, 0))  # Black background
        
        # trails
        for i in range(len(masses)):
            if len(trails[i]) > 1:
                points = [world_to_screen(p, width, height, scale, center) for p in trails[i]]
                # fading effect
                for j in range(len(points) - 1):
                    alpha = int(255 * (j / len(points)))
                    pygame.draw.line(screen, tuple(int(c * alpha / 255) for c in (255, 100, 100)), points[j], points[j+1], 1)
        
        # bodies
        for i in range(len(masses)):
            screen_pos = world_to_screen(pos[i], width, height, scale, center)
            radius = radii_game[i]
            pygame.draw.circle(screen, (255, 100, 100), screen_pos, radius)
            # outline
            pygame.draw.circle(screen, (255, 255, 255), screen_pos, radius, 1)

        
        if new_pos is not None:
            pygame.draw.line(screen, (0, 255, 0), new_pos, pygame.mouse.get_pos(), 2)
            pygame.draw.circle(screen, (0, 255, 0), new_pos, 5, 1)

        for line in lines_to_draw:
            x = (world_to_screen(np.array(line[0]), width, height, scale, center),
                 world_to_screen(np.array(line[1]), width, height, scale, center))
            pygame.draw.line(screen, (0, 255, 0), x[0], x[1], 1)

        for c in centers:
            screen_c = world_to_screen(np.array(c), width, height, scale, center)
            pygame.draw.circle(screen, (0, 0, 255), screen_c, 3)

        # Display info
        info_lines = [
            f"Step: {step}",
            f"Bodies: {len(masses)}",
            f"Time Quadtree: {time_quadtree:.4f}s" if config["algorithm"] == "barnes_hut" else "",
            f"Time Simulation: {time_simul:.4f}s",
            f"Pairwise checks: {count}",
        ]
        for i, line in enumerate(info_lines):
            if line:
                text_surf = small_font.render(line, True, (255, 255, 255))
                screen.blit(text_surf, (10, 10 + i * 18))

        pygame.display.flip()
        clock.tick(60)  # 60 FPS
    
    pygame.quit()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="galaxy", help="name of config file")
    args.add_argument("--algorithm", type=str, default="mesh_based_poisson", help="algorithm to use (naive)")

    if args.parse_args().config == 'galaxy':
        n_bodies = 100
        n_circles = 2
        masses = [1000] + [1.0]*n_bodies*n_circles
        positions = [[0.0, 0.0]] 
        for j in range(n_circles):
            positions += [[np.cos(i / n_bodies * 2 * np.pi) * 200 * (j+1), np.sin(i / n_bodies * 2 * np.pi) * 200 * (j+1)] for i in range(n_bodies)]
        velocities = [[0.0, 0.0]]
        for j in range(n_circles):
            velocities += [[-np.sin(i / n_bodies * 2 * np.pi) * 2 * (0.8 ** j) , np.cos(i / n_bodies * 2 * np.pi) * 2 * (0.8 ** j)] for i in range(n_bodies)]
        config = {
            "algorithm": args.parse_args().algorithm,
            "dt": 0.1,
            "theta": 0.5,
            "collisions": False
        }
        positions = np.array(positions)
        velocities = np.array(velocities)
        masses = np.array(masses)
    else:
        path = Path(__file__).parent / "configs" / args.parse_args().algorithm / args.parse_args().config
        if not path.exists():
            path = Path(__file__).parent / "configs" / "empty.toml"
        path = path.with_suffix('.toml').resolve()
        with path.open("rb") as f:
            config = toml.load(f)

        if config.get('random', False):
            n = config.get('n', 5)
            pos_range = config.get('position_range', [-2.0, 2.0])
            vel_range = config.get('velocity_range', [-0.5, 0.5])
            mass_range = config.get('mass_range', [0.5, 1.0])
            positions = np.random.uniform(*pos_range, (n, 2))
            velocities = np.random.uniform(*vel_range, (n, 2))
            masses = np.random.uniform(*mass_range, n)
        else:
            positions = np.array(config['positions'])
            velocities = np.array(config['velocities'])
            masses = np.array(config['masses'])

    config['algorithm'] = args.parse_args().algorithm
    dt = config.get('dt', 0.01)
    simulate_pygame(positions, velocities, masses, dt, config,
                    title="N-Body Simulation: " + args.parse_args().algorithm)
