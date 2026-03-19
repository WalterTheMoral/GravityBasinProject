import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pygame
from Util.Util import num_to_one_hot

from Classification import model
from Simulation import FixedMass, PointMass, Simulator

WIDTH, HEIGHT = 1320, 780
GRID_RECT = pygame.Rect(40, 40, 700, 700)
NORM_MIN, NORM_MAX = 0.0, 100.0

BACKGROUND = (0, 0, 0)
GRID_BG = (255, 255, 255)
GRID_LINE = (220, 220, 220)
GRID_FRAME = (160, 160, 160)
TEXT = (245, 245, 245)
PANEL_BG = (25, 25, 25)
BUTTON_BG = (55, 55, 55)
BUTTON_TEXT = (255, 255, 255)
INPUT_BG = (18, 18, 18)
INPUT_ACTIVE = (60, 60, 60)

POINT_COLORS = [
    (255, 70, 70),
    (40, 170, 255),
    (120, 255, 120),
    (255, 210, 70),
]
POINT_NAMES = ["Attractor A", "Attractor B", "Attractor C", "Mass"]

rows = np.load("basin_dataset_gpu_1E6.npz")["X"].T
answers = num_to_one_hot(3, np.load("basin_dataset_gpu_1E6.npz")["y"])
print(rows.shape)

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


class NumberInput:
    def __init__(self, rect: pygame.Rect, initial: float):
        self.rect = rect
        self.active = False
        self.text = f"{initial:.2f}"

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if self.active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.active = False
            elif event.unicode in "0123456789.-":
                self.text += event.unicode

    def value(self, fallback: float) -> float:
        try:
            return max(NORM_MIN, min(NORM_MAX, float(self.text)))
        except ValueError:
            return fallback

    def set_value(self, value: float):
        self.text = f"{value:.2f}"

    def draw(self, surf: pygame.Surface, font: pygame.font.Font):
        color = INPUT_ACTIVE if self.active else INPUT_BG
        pygame.draw.rect(surf, color, self.rect, border_radius=4)
        pygame.draw.rect(surf, (190, 190, 190), self.rect, width=1, border_radius=4)
        txt = font.render(self.text, True, TEXT)
        surf.blit(txt, (self.rect.x + 7, self.rect.y + 6))


class GravityApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Gravity Basin Predictor")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 15)

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

        bx = 780
        self.buttons = {
            "predict": Button(pygame.Rect(bx, 40, 320, 50), "Predict with model"),
            "simulate": Button(pygame.Rect(bx, 100, 320, 50), "Run simulation"),
            "reset": Button(pygame.Rect(bx, 160, 320, 50), "Reset simulation"),
            "random": Button(pygame.Rect(bx, 220, 320, 50), "Randomise points"),
        }

        self.inputs: List[Tuple[NumberInput, NumberInput]] = []
        start_y = 320
        for point in self.points:
            x_in = NumberInput(pygame.Rect(bx + 170, start_y, 110, 34), point.x)
            y_in = NumberInput(pygame.Rect(bx + 300, start_y, 110, 34), point.y)
            self.inputs.append((x_in, y_in))
            start_y += 102

    def norm_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        sx = GRID_RECT.left + int((x / 100.0) * GRID_RECT.width)
        sy = GRID_RECT.bottom - int((y / 100.0) * GRID_RECT.height)
        return sx, sy

    def screen_to_norm(self, sx: int, sy: int) -> Tuple[float, float]:
        x = (sx - GRID_RECT.left) / GRID_RECT.width * 100.0
        y = (GRID_RECT.bottom - sy) / GRID_RECT.height * 100.0
        return max(NORM_MIN, min(NORM_MAX, x)), max(NORM_MIN, min(NORM_MAX, y))

    def sync_from_inputs(self):
        for idx, point in enumerate(self.points):
            x_in, y_in = self.inputs[idx]
            point.x = x_in.value(point.x)
            point.y = y_in.value(point.y)

    def sync_to_inputs(self, idx: int):
        self.inputs[idx][0].set_value(self.points[idx].x)
        self.inputs[idx][1].set_value(self.points[idx].y)

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
        else:
            self.prediction_text = f"Prediction: {POINT_NAMES[basin]}"

    def start_simulation(self):
        p = self.points
        attractors = [
            FixedMass(p[0].x, p[0].y, 1),
            FixedMass(p[1].x, p[1].y, 1),
            FixedMass(p[2].x, p[2].y, 1),
        ]
        point_mass = PointMass(p[3].x, p[3].y, 50, 1e3)
        self.simulator = Simulator(attractors, point_mass)
        self.simulation_trace = [(point_mass.x, point_mass.y)]
        self.sim_result_text = "Simulation: running..."
        self.simulating = True

    def update_simulation(self):
        if not self.simulating or self.simulator is None:
            return

        for _ in range(180):
            done, basin = self.simulator.converged(max_distance=2, max_velocity=1)
            if done:
                self.simulating = False
                self.sim_result_text = f"Simulation: {POINT_NAMES[basin]}"
                self.points[3].x = self.simulator.point.x
                self.points[3].y = self.simulator.point.y
                self.sync_to_inputs(3)
                return
            self.simulator.update(1 / 60)
            self.simulation_trace.append((self.simulator.point.x, self.simulator.point.y))

    def reset(self):
        self.simulating = False
        self.simulator = None
        for idx, default in enumerate(self.default_points):
            self.points[idx].x = default.x
            self.points[idx].y = default.y
            self.sync_to_inputs(idx)
        self.simulation_trace.clear()
        self.prediction_text = "Prediction: -"
        self.sim_result_text = "Simulation: -"



    def _load_permutated_rows(self) -> List[List[float]]:
        csv_path = Path("Permuted Data.csv")
        if not csv_path.exists():
            return []

        rows: List[List[float]] = []
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 8:
                    continue
                try:
                    values = [float(row[i]) for i in range(8)]
                except ValueError:
                    # skip header or malformed rows
                    continue
                rows.append(values)

        return rows

    def randomise_points(self):
        self.simulating = False
        self.simulator = None

        # rows = self._load_permutated_rows()
        # if not rows:
        #     self.simulation_trace.clear()
        #     self.prediction_text = "Prediction: -"
        #     self.sim_result_text = "Simulation: unavailable (missing Permutated Data.csv)"
        #     return

        index = random.randint(0, rows.shape[1])
        print(rows.shape)
        sample = rows[:,index]

        print(sample)
        print(answers[:,index])

        # CSV format requested: [mass_x, mass_y, a1_x, a1_y, a2_x, a2_y, a3_x, a3_y, ...]
        self.points[3].x, self.points[3].y = sample[0], sample[1]
        self.points[0].x, self.points[0].y = sample[2], sample[3]
        self.points[1].x, self.points[1].y = sample[4], sample[5]
        self.points[2].x, self.points[2].y = sample[6], sample[7]

        for idx in range(4):
            self.sync_to_inputs(idx)

        self.simulation_trace.clear()
        self.prediction_text = "Prediction: -"
        self.sim_result_text = "Simulation: -"

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
        panel = pygame.Rect(760, 0, 560, HEIGHT)
        pygame.draw.rect(self.screen, PANEL_BG, panel)

        for button in self.buttons.values():
            button.draw(self.screen, self.font)

        self.screen.blit(self.font.render(self.prediction_text, True, TEXT), (780, 285))
        self.screen.blit(self.font.render(self.sim_result_text, True, TEXT), (780, 580))

        y = 312
        for idx, point in enumerate(self.points):
            title = self.font.render(point.name, True, point.color)
            self.screen.blit(title, (780, y))

            field_y = y + 8
            self.screen.blit(self.small_font.render("x:", True, TEXT), (915, field_y + 7))
            self.screen.blit(self.small_font.render("y:", True, TEXT), (1045, field_y + 7))

            x_in, y_in = self.inputs[idx]
            x_in.draw(self.screen, self.font)
            y_in.draw(self.screen, self.font)

            y += 102

    def handle_mouse_down(self, pos):
        if self.buttons["predict"].rect.collidepoint(pos):
            self.sync_from_inputs()
            self.predict()
            return
        if self.buttons["simulate"].rect.collidepoint(pos):
            self.sync_from_inputs()
            self.start_simulation()
            return
        if self.buttons["reset"].rect.collidepoint(pos):
            self.reset()
            return
        if self.buttons["random"].rect.collidepoint(pos):
            self.randomise_points()
            return

        for x_in, y_in in self.inputs:
            x_in.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": pos}))
            y_in.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"pos": pos}))

        if GRID_RECT.collidepoint(pos):
            for idx, point in enumerate(self.points):
                sx, sy = self.norm_to_screen(point.x, point.y)
                if math.dist((sx, sy), pos) < 14:
                    self.dragging_idx = idx
                    self.simulating = False
                    break

    def run(self):
        running = True
        while running:
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
                    self.sync_to_inputs(self.dragging_idx)
                elif event.type == pygame.KEYDOWN:
                    for x_in, y_in in self.inputs:
                        x_in.handle_event(event)
                        y_in.handle_event(event)

            self.update_simulation()
            if self.simulating and self.simulator is not None:
                self.points[3].x = self.simulator.point.x
                self.points[3].y = self.simulator.point.y
                self.sync_to_inputs(3)

            self.screen.fill(BACKGROUND)
            self.draw_grid()
            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    GravityApp().run()
    sys.exit(0)
