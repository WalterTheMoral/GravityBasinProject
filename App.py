# import Classification
from MachineLearningBase import *
from Simulation import *
import pygame
import sys
import math

# ==========================================
# Coordinate Transform Utilities
# ==========================================

SIM_SIZE = 100
GRID_SIZE = 650
GRID_ORIGIN = (40, 40)

def screen_to_sim(x, y):
    sim_x = (x - GRID_ORIGIN[0]) / GRID_SIZE * SIM_SIZE
    sim_y = (y - GRID_ORIGIN[1]) / GRID_SIZE * SIM_SIZE
    return sim_x, sim_y

def sim_to_screen(x, y):
    screen_x = GRID_ORIGIN[0] + (x / SIM_SIZE) * GRID_SIZE
    screen_y = GRID_ORIGIN[1] + (y / SIM_SIZE) * GRID_SIZE
    return screen_x, screen_y

# =========================================================
# === DARK MODE UI
# =========================================================

pygame.init()
WIDTH, HEIGHT = 1000, 750
GRID_SIZE = 650
GRID_ORIGIN = (40, 40)
tick = 200

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Attractor Basin Simulator")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 26)

# Dark palette
BG = (20, 22, 26)
GRID = (60, 65, 75)
TEXT = (220, 220, 220)
BUTTON = (40, 45, 55)
BUTTON_BORDER = (90, 95, 110)
GRID_BG = (255, 255, 255)
GRID_LINE = (210, 210, 210)
GRID_BORDER = (160, 160, 160)

COLORS = [(255, 80, 80), (80, 160, 255), (80, 220, 120)]
MASS_COLOR = (255, 220, 80)


# =========================================================
# === DRAGGABLE POINT
# =========================================================

class Draggable:
    def __init__(self, sim_x, sim_y, color, label):
        self.sim_x = sim_x
        self.sim_y = sim_y
        self.color = color
        self.label = label
        self.dragging = False

    def draw(self):
        screen_x, screen_y = sim_to_screen(self.sim_x, self.sim_y)
        pygame.draw.circle(screen, self.color,
                           (int(screen_x), int(screen_y)), 9)
        label = font.render(self.label, True, TEXT)
        screen.blit(label, (screen_x + 10, screen_y - 12))

    def handle_event(self, event):
        screen_x, screen_y = sim_to_screen(self.sim_x, self.sim_y)

        if event.type == pygame.MOUSEBUTTONDOWN:
            if math.hypot(event.pos[0] - screen_x,
                          event.pos[1] - screen_y) < 10:
                self.dragging = True

        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        if event.type == pygame.MOUSEMOTION and self.dragging:
            x, y = event.pos

            # Clamp inside grid
            x = max(GRID_ORIGIN[0], min(GRID_ORIGIN[0] + GRID_SIZE, x))
            y = max(GRID_ORIGIN[1], min(GRID_ORIGIN[1] + GRID_SIZE, y))

            self.sim_x, self.sim_y = screen_to_sim(x, y)

# =========================================================
# === BUTTON
# =========================================================

class Button:
    def __init__(self, rect, text):
        self.rect = pygame.Rect(rect)
        self.text = text

    def draw(self):
        pygame.draw.rect(screen, BUTTON, self.rect)
        pygame.draw.rect(screen, BUTTON_BORDER, self.rect, 2)
        t = font.render(self.text, True, TEXT)
        screen.blit(t, (self.rect.x + 10, self.rect.y + 8))

    def clicked(self, pos):
        return self.rect.collidepoint(pos)


# =========================================================
# === MODEL PLACEHOLDER
# =========================================================

def model_predict(coords):
    # Replace with ANN inference
    return 0


# =========================================================
# === INITIAL STATE
# =========================================================

# 30.0, 25.0, 72.0, 29.0, 83.0, 77.0, 0.0, 54.0
attractor_points = [
    Draggable(0, 54, COLORS[0], "A1"),
    Draggable(72, 29, COLORS[1], "A2"),
    Draggable(83, 77, COLORS[2], "A3"),
]

mass_point = Draggable(30, 25, MASS_COLOR, "Mass")

run_btn = Button((760, 300, 180, 45), "Run Simulation")
reset_btn = Button((760, 360, 180, 45), "Reset")

simulator = None
running_sim = False
prediction_text = "-"
result_text = "-"

trail = []


# =========================================================
# === MAIN LOOP
# =========================================================

while True:
    clock.tick(tick)
    screen.fill(BG)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        for p in attractor_points + [mass_point]:
            p.handle_event(event)

        if event.type == pygame.MOUSEBUTTONDOWN:
            if run_btn.clicked(event.pos):
                if run_btn.clicked(event.pos):
                    attractors = [
                        FixedMass(p.sim_x, p.sim_y, 1)
                        for p in attractor_points
                    ]

                    point_mass = PointMass(
                        mass_point.sim_x,
                        mass_point.sim_y,
                        1
                    )

                    simulator = Simulator(attractors, point_mass)

                    prediction = model_predict(
                        [(p.sim_x, p.sim_y) for p in attractor_points]
                        + [(mass_point.sim_x, mass_point.sim_y)]
                    )

                    prediction_text = f"A{prediction + 1}"
                    running_sim = True
                    trail.clear()

            if reset_btn.clicked(event.pos):
                running_sim = False
                trail.clear()
                prediction_text = "-"
                result_text = "-"

    # Run simulation
    if running_sim and simulator:
        for _ in range(54):
            simulator.update(1 / 60)
            trail.append((simulator.point.x, simulator.point.y))

        converged, basin = simulator.converged()
        if converged:
            running_sim = False
            result_text = f"A{basin+1}"

    # Draw grid
    # ==============================
    # Draw Grid (White Panel Style)
    # ==============================

    # Grid background panel
    grid_rect = pygame.Rect(
        GRID_ORIGIN[0],
        GRID_ORIGIN[1],
        GRID_SIZE,
        GRID_SIZE
    )

    pygame.draw.rect(screen, GRID_BG, grid_rect)

    # Thin grid lines
    cells = 20  # 100 sim units / 5-unit spacing = 20 lines
    spacing = GRID_SIZE / cells

    for i in range(cells + 1):
        x = GRID_ORIGIN[0] + i * spacing
        y = GRID_ORIGIN[1] + i * spacing

        pygame.draw.line(
            screen,
            GRID_LINE,
            (x, GRID_ORIGIN[1]),
            (x, GRID_ORIGIN[1] + GRID_SIZE),
            1
        )

        pygame.draw.line(
            screen,
            GRID_LINE,
            (GRID_ORIGIN[0], y),
            (GRID_ORIGIN[0] + GRID_SIZE, y),
            1
        )

    # Small frame around grid
    pygame.draw.rect(screen, GRID_BORDER, grid_rect, 2)
    # Draw trail
    for sim_pos in trail:
        sx, sy = sim_to_screen(sim_pos[0], sim_pos[1])
        pygame.draw.circle(screen, (180, 180, 180),
                           (int(sx), int(sy)), 2)

    # Draw attractors
    for p in attractor_points:
        p.draw()

    # Draw mass
    if running_sim and simulator:
        sx, sy = sim_to_screen(
            simulator.point.x,
            simulator.point.y
        )

        pygame.draw.circle(screen, MASS_COLOR,
                           (int(sx), int(sy)), 8)
    else:
        mass_point.draw()

    # Draw UI
    run_btn.draw()
    reset_btn.draw()

    pred_label = font.render(f"Prediction: {prediction_text}", True, TEXT)
    res_label = font.render(f"Result: {result_text}", True, TEXT)
    screen.blit(pred_label, (760, 450))
    screen.blit(res_label, (760, 480))

    pygame.display.flip()