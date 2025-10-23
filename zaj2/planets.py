import pygame
from pygame.math import Vector2
import sys

eps = 1e-6

class Ball:
    def __init__(self, pos, vel, mass, radius, color):
        self.pos = Vector2(pos)
        self.vel = Vector2(vel)
        self.mass = mass
        self.radius = int(radius)
        self.color = color

        self.last_pos = Vector2(pos)

    def update(self, dt, acceleration):
        if self.mass == float('inf'):
            return
        self.last_pos = self.pos.copy()
        self.vel += acceleration * dt
        self.pos += self.vel * dt

    def update_velocity(self, dt):
        self.vel = (self.pos - self.last_pos) / dt
        

    def draw(self, surf):
        pygame.draw.circle(surf, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)


class Constraint:
    def __init__(self):
        pass

    def gradient(self):
        raise NotImplementedError

class DistanceConstraint(Constraint):
    def __init__(self, body_a, body_b, distance):
        super().__init__()
        self.body_a = body_a
        self.body_b = body_b
        self.distance = distance
    
    def value(self):
        delta = self.body_b.pos - self.body_a.pos
        return delta.length() - self.distance

    def gradient(self):
        delta = self.body_b.pos - self.body_a.pos
        current_distance = delta.length()
        if current_distance == 0:
            return Vector2(0, 0), Vector2(0, 0)
        direction = delta.normalize()
        grad_a = direction
        grad_b = -direction
        return grad_a, grad_b
    
    def apply(self):
        grad_a, grad_b = self.gradient()
        w1 = 1 / self.body_a.mass if self.body_a.mass != float('inf') else 0
        w2 = 1 / self.body_b.mass if self.body_b.mass != float('inf') else 0
        w_sum = w1 + w2
        self.body_a.pos += (w1 / w_sum) * self.value() * grad_a
        self.body_b.pos += (w2 / w_sum) * self.value() * grad_b

def handle_collision(a, b, restitution):
    delta = b.pos - a.pos
    dist = delta.length()
    min_dist = a.radius + b.radius


    n = delta / dist
    penetration = max(0.0, min_dist - dist)/2

    if penetration <= 0:
        return

    a.pos -= n * penetration
    b.pos += n * penetration

    v1 = a.vel.dot(n)
    v2 = b.vel.dot(n)
    m1 = a.mass
    m2 = b.mass
    newV1 = (m1 * v1 + m2 * v2 - m2 * (v1 - v2) * restitution) / (m1 + m2)
    newV2 = (m1 * v1 + m2 * v2 - m1 * (v2 - v1) * restitution) / (m1 + m2)
    a.vel += (newV1 - v1) * n
    b.vel += (newV2 - v2) * n

def run():
    pygame.init()
    WIDTH, HEIGHT = 900, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    balls = [
        Ball(pos=(WIDTH * 0.5, HEIGHT * 0.5), vel=(0, 0), mass=float('inf'), radius=18, color=(255, 200, 0)),  # heavy central
        Ball(pos=(WIDTH * 0.5 + 150, HEIGHT * 0.5), vel=(0, 0), mass=30, radius=8, color=(255, 180, 255)),
        Ball(pos=(WIDTH * 0.5 - 150, HEIGHT * 0.5), vel=(0, 0), mass=10, radius=6, color=(100, 180, 255)),
    ]

    constraints = [
        DistanceConstraint(balls[0], balls[1], 150),
        DistanceConstraint(balls[0], balls[2], 150),
    ]

    gravity_acc = Vector2(0, 200)

    running = True
    while running:
        dt_ms = clock.tick(60)               # limit to ~60 FPS
        dt = 0.001              # convert to seconds
        substeps = 50

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False


        for _ in range(substeps):
            for i in range(len(balls)):
                balls[i].update(dt, gravity_acc)
                    
            for constraint in constraints:
                constraint.apply()

            for i in range(len(balls)):
                balls[i].update_velocity(dt)

            restitution = 0.5  # bounciness [0..1]

            for i in range(len(balls)):
                for j in range(i + 1, len(balls)):
                    a = balls[i]
                    b = balls[j]
                    handle_collision(a, b, restitution)

        # Draw
        screen.fill((10, 10, 30))
        for b in balls:
            b.draw(screen)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run()