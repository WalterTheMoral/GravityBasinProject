import csv
import os.path
import random
import time
import itertools
import multiprocessing as mp

from Simulation import *

def random_coordinates():
    x = random.uniform(0, 100)
    y = random.uniform(0, 100)
    return x, y

def generate_sample(_):
    attractors = [
        FixedMass(*random_coordinates(), 1)
        for _ in range(3)
    ]
    mass_coords = random_coordinates()
    point_mass = PointMass(*mass_coords, 50)
    sim = Simulator(attractors, point_mass)

    start = time.time()
    convergence_point = sim.converge_to_which_basin()
    elapsed = start - time.time()


    return [
        *mass_coords,
        attractors[0].point[0], attractors[0].point[1],
        attractors[1].point[0], attractors[1].point[1],
        attractors[2].point[0], attractors[2].point[1],
        convergence_point,
        elapsed
    ]


if __name__ == "__main__":
    fieldnames = [
        "Point Mass X", "Point Mass Y",
        "Fixed Point 1 X", "Fixed Point 1 Y",
        "Fixed Point 2 X", "Fixed Point 2 Y",
        "Fixed Point 3 X", "Fixed Point 3 Y",
        "Convergence Point", "Convergence Time"
    ]

    num_samples = 300_000
    num_workers = mp.cpu_count() - 2  # use all cores

    print("Start")

    with mp.Pool(num_workers) as pool:
        results = pool.imap_unordered(generate_sample, range(num_samples), chunksize=100)

        with open("database_v3.csv", "a", newline="") as f:
            writer = csv.writer(f)

            i = 0
            for row in results:
                writer.writerow(row)
                i += 1

                if i % 100 == 0:
                    print(i)