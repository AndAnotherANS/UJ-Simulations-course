import math
import random
import sys
import pygame
import numpy as np

# pygame constants
WIDTH, HEIGHT = 800, 600
BG_COLOR = (25, 25, 30)
FPS = 120

# simulation constants
NUM_BALLS = 10
MIN_RADIUS, MAX_RADIUS = 10, 25
MIN_SPEED, MAX_SPEED = 80, 240  # pixels/second

ELASTICITY = 1.0  # 1.0 = perfectly elastic

class Ball:
    def __init__(self, x, v, color):
        self.x = np.array(x, dtype=float)
        self.v = np.array(v, dtype=float)
        self.color = color
        self.r = 10

    def update(self, dt):
        self.x += self.v * dt


    def bounce_walls(self, w, h, e=ELASTICITY):
        # Left
        if self.x[0] - self.r < 0:
            self.x[0] = self.r
            self.v[0] = -self.v[0] * e
        # Right
        if self.x[0] + self.r > w:
            self.x[0] = w - self.r
            self.v[0] = -self.v[0] * e
        # Top
        if self.x[1] - self.r < 0:
            self.x[1] = self.r
            self.v[1] = -self.v[1] * e
        # Bottom
        if self.x[1] + self.r > h:
            self.x[1] = h - self.r
            self.v[1] = -self.v[1] * e

    def draw(self, surf):
        pygame.draw.circle(surf, self.color, (int(self.x[0]), int(self.x[1])), self.r)

def random_ball(existing, w, h, max_tries=100):
    for _ in range(max_tries):
        r = random.randint(MIN_RADIUS, MAX_RADIUS)
        x = random.uniform(r, w - r)
        y = random.uniform(r, h - r)
        speed = random.uniform(MIN_SPEED, MAX_SPEED)
        angle = random.uniform(0, 2 * math.pi)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        color = tuple(random.randint(60, 255) for _ in range(3))
        b = Ball(np.array([x, y]), np.array([vx, vy]), color)
        # Avoid initial overlap
        if all((dist(b, other) > b.r + other.r) for other in existing):
            return b
    return None  # fallback if cannot place

def dist(a, b):
    dx = a.x - b.x
    return np.linalg.norm(dx)

def resolve_collisions(balls, e=ELASTICITY):
    n = len(balls)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = balls[i], balls[j]
            d = dist(a, b)
            min_dist = a.r + b.r

            dx = a.x - b.x
            norm_dx = dx / np.linalg.norm(dx)

            if d < min_dist:
                
                overlap = (min_dist - d)

                a.x += norm_dx * overlap / 2
                b.x -= norm_dx * overlap / 2

                a.v = a.v - np.dot(a.v - b.v, a.x - b.x) / np.dot(a.x - b.x, a.x - b.x) * (a.x - b.x) * e
                b.v = b.v - np.dot(b.v - a.v, b.x - a.x) / np.dot(b.x - a.x, b.x - a.x) * (b.x - a.x) * e

def main():
    pygame.init()
    pygame.display.set_caption("Bouncing Balls - Pygame")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    balls = []
    for _ in range(NUM_BALLS):
        b = random_ball(balls, WIDTH, HEIGHT)
        if b:
            balls.append(b)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    balls.clear()
                elif event.key == pygame.K_a:
                    b = random_ball(balls, WIDTH, HEIGHT)
                    if b:
                        balls.append(b)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                r = random.randint(MIN_RADIUS, MAX_RADIUS)
                # Clamp inside box
                mx = max(r, min(WIDTH - r, mx))
                my = max(r, min(HEIGHT - r, my))
                speed = random.uniform(MIN_SPEED, MAX_SPEED)
                angle = random.uniform(0, 2 * math.pi)
                vx = speed * math.cos(angle)
                vy = speed * math.sin(angle)
                color = tuple(random.randint(60, 255) for _ in range(3))
                balls.append(Ball(np.array([mx, my]), np.array([vx, vy]), color))

        # Update physics
        for b in balls:
            b.update(dt)
            b.bounce_walls(WIDTH, HEIGHT)

        resolve_collisions(balls)

        # Render
        screen.fill(BG_COLOR)
        for b in balls:
            b.draw(screen)
        # UI hint
        draw_text(screen, "LMB: add ball  |  A: add  |  R: reset  |  ESC/Q: quit", 14, (200, 200, 210), (10, 10))

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)

def draw_text(surf, text, size, color, pos):
    font = pygame.font.SysFont(None, size)
    surf.blit(font.render(text, True, color), pos)

if __name__ == "__main__":
    main()