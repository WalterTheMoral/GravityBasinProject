import csv
import random
import time
import multiprocessing as mp

from Simulation import *

def random_coordinates():
    x = random.uniform(0, 100)
    y = random.uniform(0, 100)
    return x, y

def generate_sample(_):
    attractors = [
        FixedMass(25, 75, 1),
        FixedMass(70, 20, 1),
        FixedMass(65, 75, 1)
    ]
    point_mass = PointMass(*random_coordinates(), 50)
    sim = Simulator(attractors, point_mass)

    start = time.time()
    convergence_point = sim.converge_to_which_basin()
    elapsed = start - time.time()

    return [
        point_mass.point[0], point_mass.point[1],
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

    num_samples = 100_000
    num_workers = mp.cpu_count()  # use all cores

    with mp.Pool(num_workers) as pool:
        results = pool.imap_unordered(generate_sample, range(num_samples), chunksize=100)

        with open("fixed_attractors.csv", "a", newline="") as f:
            writer = csv.writer(f)

            for i, row in enumerate(results):
                writer.writerow(row)

                if i % 1000 == 0:
                    print(i)