import numpy as np
from dataclasses import dataclass
from typing import Tuple
import time

G = 1.0  # gravitational constant

@dataclass
class QuadTreeNode:
    """Represents a node in the quadtree"""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    mass: float = 0.0
    center_x: float = 0.0
    center_y: float = 0.0
    children: list = None
    particle_idx: int = -1
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
def build_quadtree(positions: np.ndarray, masses: np.ndarray):
    x_min, x_max = positions[:, 0].min() - 1, positions[:, 0].max() + 1
    y_min, y_max = positions[:, 1].min() - 1, positions[:, 1].max() + 1
    
    root = QuadTreeNode(x_min, x_max, y_min, y_max)
    
    for idx, (x, y) in enumerate(positions):
        insert_particle(root, x, y, idx, positions, masses)
    
    return root

def insert_particle(node: QuadTreeNode, x: float, y: float, idx: int, positions: np.ndarray, masses: int):
    node.mass += masses[idx]
    node.center_x = (node.center_x * (node.mass - masses[idx]) + x * masses[idx]) / node.mass
    node.center_y = (node.center_y * (node.mass - masses[idx]) + y * masses[idx]) / node.mass

    if len(node.children) == 0 and node.particle_idx == -1:
        node.particle_idx = idx
    elif len(node.children) == 0:
        split_node(node, positions, masses)
        insert_particle(node, x, y, idx, positions, masses)
    else:
        quadrant = get_quadrant(node, x, y)
        insert_particle(node.children[quadrant], x, y, idx, positions, masses)

def split_node(node: QuadTreeNode, positions: np.ndarray, masses: np.ndarray):
    mid_x = (node.x_min + node.x_max) / 2
    mid_y = (node.y_min + node.y_max) / 2
    
    node.children = [
        QuadTreeNode(node.x_min, mid_x, mid_y, node.y_max),  # up-left
        QuadTreeNode(mid_x, node.x_max, mid_y, node.y_max),  # up-right
        QuadTreeNode(node.x_min, mid_x, node.y_min, mid_y),  # down-left
        QuadTreeNode(mid_x, node.x_max, node.y_min, mid_y),  # down-right
    ]
    
    #insert existing particle
    if node.particle_idx != -1:
        existing_idx = node.particle_idx
        existing_x, existing_y = positions[existing_idx]
        quadrant = get_quadrant(node, existing_x, existing_y)
        node.children[quadrant].mass = masses[existing_idx]
        node.children[quadrant].center_x = existing_x
        node.children[quadrant].center_y = existing_y
        node.children[quadrant].particle_idx = existing_idx
        node.particle_idx = -1

def get_quadrant(node: QuadTreeNode, x: float, y: float):
    mid_x = (node.x_min + node.x_max) / 2
    mid_y = (node.y_min + node.y_max) / 2
    if x < mid_x:
        return 0 if y >= mid_y else 2
    return 1 if y >= mid_y else 3

def get_lines_to_draw(node: QuadTreeNode, lines: list[tuple[(int,int),(int,int)]]):
    lines.append(((int(node.x_min), int(node.y_min)), (int(node.x_max), int(node.y_min))))
    lines.append(((int(node.x_max), int(node.y_min)), (int(node.x_max), int(node.y_max))))
    lines.append(((int(node.x_max), int(node.y_max)), (int(node.x_min), int(node.y_max))))
    lines.append(((int(node.x_min), int(node.y_max)), (int(node.x_min), int(node.y_min))))

def get_centers_to_draw(node: QuadTreeNode, centers: list[tuple[int,int]]):
    if node.mass > 0 and not node.children:
        centers.append(node.mass)
    for child in node.children:
        get_centers_to_draw(child, centers)

def calculate_acceleration(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray, dt: float, theta: float = 0.8):
    n = len(positions)
    accelerations = np.zeros((n, 2))

    if n == 0:
        return positions, velocities, [], 0, 0, 0
    if n == 1:
        positions += velocities * dt
        return positions, velocities, [], 0, 0, 0
    
    # Build quadtree
    time_quadtree = time.time()
    root = build_quadtree(positions, masses)
    time_quadtree = time.time() - time_quadtree
    output_lines = []
    get_lines_to_draw(root, output_lines)

    # Simulate
    time_simul = time.time()
    count = [0]
    for i in range(n):
        ax, ay = calculate_force(positions[i], root, theta, positions, output_lines, count)
        accelerations[i] = [ax, ay]
    
    velocities = velocities + accelerations * dt
    positions = positions + velocities * dt
    time_simul = time.time() - time_simul
    

    return positions, velocities, output_lines, time_quadtree, time_simul, count[0]

def calculate_force(pos: np.ndarray, node: QuadTreeNode, theta: float, 
                    positions: np.ndarray, output_lines: list[tuple[(int,int),(int,int)]], count: list[int]):
    """Calculate force on a particle from a node"""
    count[0] += 1
    if node.mass == 0:
        return 0.0, 0.0
    
    r_vec = np.array([node.center_x, node.center_y]) - pos
    r_mag = np.linalg.norm(r_vec)
    
    if r_mag < 1e-10:
        return 0.0, 0.0
    
    size = max(node.x_max - node.x_min, node.y_max - node.y_min)
    
    if size / r_mag < theta or len(node.children) == 0:
        # treat as a single body
        get_lines_to_draw(node, output_lines)
        force = G * node.mass * r_vec / r_mag**3
        return force[0], force[1]
    else:
        # recursion otherwise
        ax, ay = 0.0, 0.0
        for child in node.children:
            fx, fy = calculate_force(pos, child, theta, positions, output_lines, count)
            ax += fx
            ay += fy
        return ax, ay
