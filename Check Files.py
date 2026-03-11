import csv
import math

import DatabaseGeneration


def euclidean_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def process_csv(filepath):
    with open(filepath, newline="") as f:
        reader = csv.reader(f)

        for row_idx, row in enumerate(reader):
            if "Point" in row[0]:
                print(row)
                continue
            # Convert to floats (ignore label)
            values = list(map(float, row[:8]))

            # Extract attractors
            a1 = values[6:8]
            a2 = values[2:4]
            a3 = values[4:6]

            # Extract mass
            mass = values[0:2]

            # Compute distances
            distances = [
                euclidean_distance(*mass, *a1),
                euclidean_distance(*mass, *a2),
                euclidean_distance(*mass, *a3),
            ]

            smallest = min(distances)

            print(f"Row {row_idx}: {smallest}")


if __name__ == "__main__":
    # process_csv("Permuted Data.csv")
    print(DatabaseGeneration.generate_sample(0))