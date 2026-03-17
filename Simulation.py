import math
from math import atan2
from typing import List, Tuple
import numpy as np
import time

class Vector:
    def __init__(self, x_or_r: float = 0, y_or_theta: float = 0, cartesian=True) -> None:
        if cartesian:
            self.x = x_or_r
            self.y = y_or_theta

        else:
            self.x = x_or_r * np.cos(y_or_theta)
            self.y = x_or_r * np.sin(y_or_theta)


    def __add__(self, other): # Typing
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, other: float):
        return Vector(self.x * other, self.y * other)

    def __truediv__(self, other: float):
        return Vector(self.x / other, self.y / other)

    def cartesian_form(self) -> Tuple[float, float]:
        return self.x, self.y

    def polar_form(self) -> Tuple[float, float]:
        return float(np.linalg.norm((self.x, self.y))), math.atan2(self.y, self.x)

    def __str__(self) -> str:
        return f"{str(self.cartesian_form())}"


class FixedMass:
    def __init__(self, x: float, y: float, mass: float = 10) -> None:
        self.x = x
        self.y = y
        self.mass = mass

        self.point = (x, y)


class PointMass(FixedMass):
    def __init__(self, x: float, y: float, mass: float = 10, g_constant: float = 1) -> None:
        super().__init__(x, y, mass)
        self.velocity = Vector()
        self.g_constant = g_constant

    def gravitational_force(self, attractors: List[FixedMass], epsilon: float = 8) -> Vector:
        F = Vector()

        for attractor in attractors:
            distance = np.linalg.norm((self.x - attractor.x, self.y - attractor.y))
            f_i = float((self.g_constant * self.mass * attractor.mass) / (distance + epsilon)**2)
            f_gi = Vector(f_i, atan2(-self.y + attractor.y, -self.x + attractor.x), False)

            F += f_gi

        return F

    def friction_force(self, coefficient: float = 0.01):
        return self.velocity * -coefficient


    def update(self, force: Vector, dt: float = 1/60):
        self.velocity += (force / self.mass)  * dt
        vx, vy = self.velocity.cartesian_form()

        self.x += vx * dt
        self.y += vy * dt

        self.point = self.x, self.y

    def __str__(self):
        return f"Fixed Mass: ({self.x}, {self.y})"


class Simulator:
    def __init__(self, attractors: List[FixedMass], point: PointMass):
        self.attractors = attractors
        self.point = point

    def update(self, dt: float = 1/60):
        force = self.point.gravitational_force(self.attractors) + self.point.friction_force(0.05)
        self.point.update(force, dt)

    def converged_to_fixed_mass(self, fixed_mass: FixedMass | int,
                  max_distance: float = 2, max_velocity: float = 1) -> bool:
        if type(fixed_mass) == int:
            fixed_mass = attractors[fixed_mass]

        distance = np.linalg.norm((fixed_mass.x - self.point.x, fixed_mass.y - self.point.y))
        if distance < max_distance and self.point.velocity.polar_form()[0] < max_velocity:
            return True

        return False

    def converged(self, max_distance: float = 2, max_velocity: float = 3) -> Tuple[bool, int | None]:
        for n, attractor in enumerate(self.attractors):
            if self.converged_to_fixed_mass(attractor, max_distance, max_velocity):
                return True, n

        return False, None

    def converge_to_which_basin(self, dt: float = 1/60, max_distance: float = 2, max_velocity: float = 1) -> int:
        while True:
            converged, basin = self.converged(max_distance, max_velocity)
            if converged:
                return basin
            self.update(dt)


if __name__ == "__main__":
    M = 1
    m = 1
    g_constant = 1e3

    attractors = [FixedMass(10, 60, M), FixedMass(40, 20, M), FixedMass(50, 70, M)]
    sim = Simulator(attractors, PointMass(000, 100, 1, 10))

    start = time.time()
    print(sim.converge_to_which_basin())
    print(time.time() - start)