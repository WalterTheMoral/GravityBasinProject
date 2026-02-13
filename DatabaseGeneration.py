import csv
import random
import time
import multiprocessing as mp

from Simulation import *

def random_coordinates():
    x = random.uniform(0, 100)
    y = random.uniform(0, 100)
    return x, y

def generate_samples(_):
    attractors = [
        FixedMass(25, 75, 1),
        FixedMass(70, 20, 1),
        FixedMass(65, 75, 1)
    ]
    point_mass = PointMass(*random_coordinates(), 50)
    sim = Simulator(attractors, point_mass)

    start = time.time()
    sim.converge_to_which_basin()
    elapsed = start - time.time()

    return [
        point_mass.point[0],
        point_mass.point[1],
        attractors[0].point[0], attractors[0].point[1],
        attractors[1].point[0], attractors[1].point[1],
        attractors[2].point[0], attractors[2].point[1],
        convergence_point,
        elapsed
    ]

full_start = time.time()
with open('fixed_attractors.csv', 'a', newline='') as database:
    fieldnames = ["Point Mass X", "Point Mass Y",
                "Fixed Point 1 X", "Fixed Point 1 Y",
                "Fixed Point 2 X", "Fixed Point 2 Y",
                "Fixed Point 3 X", "Fixed Point 3 Y",
                "Convergence Point", "Convergence Time"
    ]
    point_fields = fieldnames[0:2]
    attractor_fields = [
        fieldnames[2 + 2 * i: 4 + 2 * i]
        for i in range(3)
    ]

    writer = csv.DictWriter(database, fieldnames=fieldnames)
    writer.writeheader()

    # for i in range(100):

    attractors = [
        FixedMass(25, 75, 1),
        FixedMass(70, 20, 1),
        FixedMass(65, 75, 1)
    ]

    i = -1
    while True:
        i+=1
        point_mass = PointMass(*random_coordinates(), 50)
        # attractors = [FixedMass(*random_coordinates(), 1) for _ in range(3)]
        sim = Simulator(attractors, point_mass)

        point_dict = {k: v for k, v in zip(point_fields, point_mass.point)}
        attractor_dicts = {
            k: v
            for fields, attractor in zip(attractor_fields, attractors)
            for k, v in zip(fields, attractor.point)
        }

        start = time.time()
        convergence_point = sim.converge_to_which_basin()
        elapsed = time.time() - start

        convergence_and_time_dict = {"Convergence Point": convergence_point, "Convergence Time": elapsed}

        writer.writerow(point_dict | attractor_dicts | convergence_and_time_dict)

        print(i)

print(time.time() - full_start)
