import math
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pygame

from Classification import model
from Simulation import FixedMass, PointMass, Simulator

WIDTH, HEIGHT = 1150, 780
GRID_RECT = pygame.Rect(40, 40, 700, 700)
NORM_MIN, NORM_MAX = 0.0, 100.0

BACKGROUND = (25, 25, 25)
GRID_BG = (255, 255, 255)
GRID_LINE = (220, 220, 220)
GRID_FRAME = (160, 160, 160)
TEXT = (245, 245, 245)
PANEL_BG = (25, 25, 25)
BUTTON_BG = (55, 55, 55)
BUTTON_TEXT = (255, 255, 255)

POINT_COLORS = [
    (255, 70, 70),
    (120, 255, 120),
    (40, 170, 255),
    (255, 210, 70),
]
POINT_NAMES = ["Red Planet", "Green Planet", "Blue Planet", "Asteroid"]


@dataclass
class MassPoint:
    x: float
    y: float
    color: Tuple[int, int, int]
    name: str


class Button:
    def __init__(self, rect: pygame.Rect, label: str):
        self.rect = rect
        self.label = label

    def draw(self, surf: pygame.Surface, font: pygame.font.Font):
        pygame.draw.rect(surf, BUTTON_BG, self.rect, border_radius=8)
        pygame.draw.rect(surf, (220, 220, 220), self.rect, width=2, border_radius=8)
        text = font.render(self.label, True, BUTTON_TEXT)
        surf.blit(text, text.get_rect(center=self.rect.center))


class GravityApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Gravity Basin Predictor")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.label_font = pygame.font.SysFont("consolas", 22)
        self.small_font = pygame.font.SysFont("consolas", 16)

        self.default_points = [
            MassPoint(20, 20, POINT_COLORS[0], POINT_NAMES[0]),
            MassPoint(80, 20, POINT_COLORS[1], POINT_NAMES[1]),
            MassPoint(50, 80, POINT_COLORS[2], POINT_NAMES[2]),
            MassPoint(50, 50, POINT_COLORS[3], POINT_NAMES[3]),
        ]
        self.points = [MassPoint(p.x, p.y, p.color, p.name) for p in self.default_points]
        self.dragging_idx: Optional[int] = None

        self.simulator: Optional[Simulator] = None
        self.simulating = False
        self.simulation_trace: List[Tuple[float, float]] = []

        self.prediction_text = "Prediction: -"
        self.sim_result_text = "Simulation: -"
        self.prediction_color = TEXT
        self.sim_result_color = TEXT

        bx = 780
        self.buttons = {
            "simulate": Button(pygame.Rect(bx, 40, 320, 50), "Run Simulation"),
            "stop": Button(pygame.Rect(bx, 100, 320, 50), "Stop Simulation"),
            "random": Button(pygame.Rect(bx, 160, 320, 50), "Randomise Points"),
            "predict": Button(pygame.Rect(bx, 220, 320, 50), "Predict Model"),
        }

    def norm_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        sx = GRID_RECT.left + int((x / 100.0) * GRID_RECT.width)
        sy = GRID_RECT.bottom - int((y / 100.0) * GRID_RECT.height)
        return sx, sy

    def screen_to_norm(self, sx: int, sy: int) -> Tuple[float, float]:
        x = (sx - GRID_RECT.left) / GRID_RECT.width * 100.0
        y = (GRID_RECT.bottom - sy) / GRID_RECT.height * 100.0
        return max(NORM_MIN, min(NORM_MAX, x)), max(NORM_MIN, min(NORM_MAX, y))

    def feature_vector(self) -> List[float]:
        p = self.points
        return [p[3].x, p[3].y, p[0].x, p[0].y, p[1].x, p[1].y, p[2].x, p[2].y]

    def model_predict(self) -> Optional[int]:
        x = np.array(self.feature_vector(), dtype=float).reshape(8, 1)
        x = 2 * (x / 100.0) - 1

        try:
            prediction = model.predict(x)
        except Exception:
            return None

        prediction = np.asarray(prediction)
        if prediction.ndim != 2 or prediction.shape[0] < 3:
            return None

        return int(np.argmax(prediction, axis=0)[0])

    def predict(self):
        basin = self.model_predict()
        if basin is None or basin not in (0, 1, 2):
            self.prediction_text = "Prediction: unavailable"
            self.prediction_color = TEXT
        else:
            self.prediction_text = f"Prediction: {POINT_NAMES[basin]}"
            self.prediction_color = POINT_COLORS[basin]

    def start_simulation(self):
        p = self.points
        attractors = [
            FixedMass(p[0].x, p[0].y, 1),
            FixedMass(p[1].x, p[1].y, 1),
            FixedMass(p[2].x, p[2].y, 1),
        ]
        point_mass = PointMass(p[3].x, p[3].y, 50, 1)
        self.simulator = Simulator(attractors, point_mass)
        self.simulation_trace = [(point_mass.x, point_mass.y)]
        self.sim_result_text = "Simulation: running..."
        self.sim_result_color = TEXT
        self.simulating = True

    def update_simulation(self):
        if not self.simulating or self.simulator is None:
            return

        for _ in range(180):
            done, basin = self.simulator.converged(max_distance=2, max_velocity=1)
            if done:
                self.simulating = False
                self.sim_result_text = f"Simulation: {POINT_NAMES[basin]}"
                self.sim_result_color = POINT_COLORS[basin]
                self.points[3].x = self.simulator.point.x
                self.points[3].y = self.simulator.point.y
                return
            self.simulator.update(1 / 60)
            self.simulation_trace.append((self.simulator.point.x, self.simulator.point.y))

    def randomise_points(self):
        for point in self.points:
            point.x = random.uniform(NORM_MIN, NORM_MAX)
            point.y = random.uniform(NORM_MIN, NORM_MAX)

        self.simulation_trace.clear()
        self.sim_result_text = "Simulation: -"
        self.sim_result_color = TEXT
        self.predict()
        self.simulating = False


    def stop_simulation(self):
        self.simulating = False
        self.simulator = None
        self.sim_result_text = "Simulation: stopped"
        self.sim_result_color = TEXT

    def on_positions_changed(self):
        self.simulation_trace.clear()
        self.sim_result_text = "Simulation: -"
        self.sim_result_color = TEXT
        self.predict()
        self.simulating = False

    def draw_grid(self):
        pygame.draw.rect(self.screen, GRID_BG, GRID_RECT)
        step = GRID_RECT.width // 10
        for i in range(11):
            x = GRID_RECT.left + i * step
            y = GRID_RECT.top + i * step
            pygame.draw.line(self.screen, GRID_LINE, (x, GRID_RECT.top), (x, GRID_RECT.bottom), 1)
            pygame.draw.line(self.screen, GRID_LINE, (GRID_RECT.left, y), (GRID_RECT.right, y), 1)
        pygame.draw.rect(self.screen, GRID_FRAME, GRID_RECT, width=3)

        if len(self.simulation_trace) > 1:
            trace_points = [self.norm_to_screen(x, y) for x, y in self.simulation_trace]
            pygame.draw.lines(self.screen, (0, 0, 0), False, trace_points, 2)

        for point in self.points:
            sx, sy = self.norm_to_screen(point.x, point.y)
            pygame.draw.circle(self.screen, point.color, (sx, sy), 10)
            pygame.draw.circle(self.screen, (0, 0, 0), (sx, sy), 10, 2)

    def draw_panel(self):
        panel = pygame.Rect(760, 0, WIDTH - 760, HEIGHT)
        pygame.draw.rect(self.screen, PANEL_BG, panel)

        for button in self.buttons.values():
            button.draw(self.screen, self.font)

        self.screen.blit(self.font.render(self.prediction_text, True, self.prediction_color), (780, 315))
        self.screen.blit(self.font.render(self.sim_result_text, True, self.sim_result_color), (780, 340))

        y = 390
        line_gap = 60
        for idx, point in enumerate(self.points):
            if idx < 3:
                label = f"{point.name}: ({point.x:.2f}, {point.y:.2f})"
                self.screen.blit(self.label_font.render(label, True, point.color), (780, y))
            else:
                label = f"{point.name}: ({point.x:.2f}, {point.y:.2f})"
                self.screen.blit(self.label_font.render(label, True, point.color), (780, y))
                self.screen.blit(self.small_font.render("[drag to move]", True, point.color), (780, y + 34))
            y += line_gap

    def handle_mouse_down(self, pos):
        if self.buttons["simulate"].rect.collidepoint(pos):
            self.start_simulation()
            return
        if self.buttons["stop"].rect.collidepoint(pos):
            self.stop_simulation()
            return
        if self.buttons["random"].rect.collidepoint(pos):
            self.randomise_points()
            return
        if self.buttons["predict"].rect.collidepoint(pos):
            self.predict()
            return

        if GRID_RECT.collidepoint(pos):
            for idx, point in enumerate(self.points):
                sx, sy = self.norm_to_screen(point.x, point.y)
                if math.dist((sx, sy), pos) < 14:
                    self.dragging_idx = idx
                    break

    def run(self):
        running = True
        while running:
            changed_by_drag = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_mouse_down(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.dragging_idx = None
                elif event.type == pygame.MOUSEMOTION and self.dragging_idx is not None:
                    nx, ny = self.screen_to_norm(*event.pos)
                    self.points[self.dragging_idx].x = nx
                    self.points[self.dragging_idx].y = ny
                    changed_by_drag = True

            if changed_by_drag:
                self.on_positions_changed()

            self.update_simulation()
            if self.simulating and self.simulator is not None:
                self.points[3].x = self.simulator.point.x
                self.points[3].y = self.simulator.point.y

            self.screen.fill(BACKGROUND)
            self.draw_grid()
            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    GravityApp().run()
    sys.exit(0)