import pygame
import math

# Initialize Pygame
pygame.init()

# Set up display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Rotating Triangle with Ball")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Physics parameters
gravity = 0.1
damping = 0.8

# Triangle definition relative to center point
triangle_points = [
    (40, 0),  # Top vertex
    (-30, -60), # Bottom-left vertex
    (30, -60)   # Bottom-right vertex
]

class Ball:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 3
        self.vy = 2

# Initialize ball at a certain position
ball = Ball(400, 300)

# Triangle center
triangle_center_x = width // 2
triangle_center_y = height // 2

angle = 0.0

# Clock for timing
clock = pygame.time.Clock()

running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update angle
    angle += 1  # Increase angle each frame to rotate the triangle
    if angle >= 360:
        angle = 0

    # Rotate triangle points
    rotated_points = []
    for point in triangle_points:
        x, y = point
        # Calculate position relative to center
        new_x = x * math.cos(math.radians(angle)) - y * math.sin(math.radians(angle))
        new_y = x * math.sin(math.radians(angle)) + y * math.cos(math.radians(angle))
        screen_x = int(triangle_center_x + new_x)
        screen_y = int(triangle_center_y + new_y)
        rotated_points.append((screen_x, screen_y))

    # Update ball position
    ball.x += ball.vx
    ball.y += ball.vy

    # Apply gravity
    ball.vy += gravity

    # Ball collision with window boundaries (uncomment if needed)
    # if ball.x < 0 or ball.x > width:
    #     ball.vx *= -damping
    # if ball.y < 0 or ball.y > height:
    #     ball.vy *= -damping

    # Collision detection with triangle edges
    for i in range(3):
        p1 = rotated_points[i]
        p2 = rotated_points[(i + 1) % 3]

        x1, y1 = p1
        x2, y2 = p2

        ball_x, ball_y = int(ball.x), int(ball.y)

        dx = x2 - x1
        dy = y2 - y1
        d = dx * (ball_y - y1) - dy * (ball_x - x1)
        if d <= 0:
            continue

        # Check collision with line segment
        t = ((ball_x - x1) * dx + (ball_y - y1) * dy) / (dx**2 + dy**2)
        if t < 0 or t > 1:
            continue

        dist_sq = (ball.x - x1)**2 + (ball.y - y1)**2
        if dist_sq >= 50:  # Radius of ball is approximately 5 units
            continue

        # Collision detected, reverse velocity based on normal vector
        nx = dy
        ny = -dx
        len_n = math.sqrt(nx**2 + ny**2)
        nx /= len_n
        ny /= len_n

        dot = ball.vx * nx + ball.vy * ny
        ball.vx -= 2 * dot * nx
        ball.vy -= 2 * dot * ny
        ball.x += nx * 50
        ball.y += ny * 50

    # Draw
    screen.fill(WHITE)

    # Draw triangle
    pygame.draw.lines(screen, BLUE, True, rotated_points, 2)

    # Draw ball
    pygame.draw.circle(screen, RED, (int(ball.x), int(ball.y)), 5)

    # Update display
    pygame.display.flip()

    # Control frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()
