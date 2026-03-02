import pygame
import random
from Simulation import *
import csv

def mult(tuple):
    return (tuple[0] * 10, tuple[1] * 10)

pygame.init()
screen = pygame.display.set_mode([1000, 1000])
clock = pygame.time.Clock()
clock.tick(60)

# attractors = [FixedMass(400, 300, 1000000), FixedMass(700, 100, 100000), FixedMass(50, 700, 100)]
M = 1
m = 1
g_constant = 1e3

I = 6
J = 6

# def random_attractor():
#     x = random.randint(0, 100)
#     y = random.randint(0, 100)
#     print(f"{x}, {y}")
#
#     return FixedMass(x, y, M)
#
# attractors = [random_attractor() for _ in range(3)]
# sims = [Simulator(attractors, PointMass(random.randint(0, 100), random.randint(0, 100), 50))]

with open("database.csv", "r") as points:
    pointReader = csv.reader(points)
    data = [float(x) for x in list(pointReader)[28]]
    print(data)
    point = PointMass(data[0], data[1], 50)
    attractors = [FixedMass(data[2*i+2], data[2*i+3], 1) for i in range(3)]
    print([attractor.point for attractor in attractors])
    sims = [Simulator(attractors, point)]

paths = [[] for _ in sims]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            sims.append(
                Simulator(attractors,
                          PointMass(x/10, y/10, m, g_constant=g_constant))
            )
            paths.append([])

    screen.fill((255, 255, 255))

    for sim in sims:
        for _ in range(60):
            sim.update(1/60)
        if sim.converged()[0]:
            print(f"{sims.index(sim)} converged to {sim.converged()[1]}")
            sims.remove(sim)

    for attractor in attractors:
        pygame.draw.circle(screen, (0,0,0), mult(attractor.point), 20)
    for i, sim in enumerate(sims):
        pygame.draw.circle(screen, (0, 0, 255), mult(sim.point.point), 10)
        # pygame.draw.circle(screen, (i * 255//len(sims), 0, 0), mult(sim.point.point), 10)
        # print(sim.point.point)
        paths[i].append(sim.point.point)

    for i, path in enumerate(paths):
        for point in path:
            pygame.draw.circle(screen, (i * 255//len(sims), 0, 0), mult(point), 1)


    # Flip the display to bring our drawings to the screen
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()
