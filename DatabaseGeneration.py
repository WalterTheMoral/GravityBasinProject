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

def generate_sample_with_permutations(_):

    attractors = [
        FixedMass(*random_coordinates(), 1)
        for _ in range(3)
    ]

    point_mass = PointMass(*random_coordinates(), 50)
    sim = Simulator(attractors, point_mass)

    start = time.time()
    convergence_point = sim.converge_to_which_basin()
    elapsed = time.time() - start

    results = []

    for perm in itertools.permutations(range(3)):

        # perm maps new_index -> old_index
        permuted_attractors = [attractors[i] for i in perm]

        # find where original convergence index moved
        new_convergence_index = perm.index(convergence_point)

        row = [
            point_mass.point[0], point_mass.point[1],
            permuted_attractors[0].point[0], permuted_attractors[0].point[1],
            permuted_attractors[1].point[0], permuted_attractors[1].point[1],
            permuted_attractors[2].point[0], permuted_attractors[2].point[1],
            new_convergence_index,
            elapsed
        ]

        results.append(row)

    return results


if __name__ == "__main__":
    fieldnames = [
        "Point Mass X", "Point Mass Y",
        "Fixed Point 1 X", "Fixed Point 1 Y",
        "Fixed Point 2 X", "Fixed Point 2 Y",
        "Fixed Point 3 X", "Fixed Point 3 Y",
        "Convergence Point", "Convergence Time"
    ]

    num_samples = 1_000_000
    num_workers = mp.cpu_count() - 3  # use all cores

    with mp.Pool(num_workers) as pool:
        results = pool.imap_unordered(generate_sample_with_permutations, range(num_samples), chunksize=100)

        with open("permuted_database.csv", "a", newline="") as f:
            writer = csv.writer(f)

            # if os.path.getsize(f.name) == 0:
            #     writer.writeheader()

            i = 0
            for rows in results:
                for row in rows:
                    writer.writerow(row)
                    i += 1

                if i % 100 == 0:
                    print(i)