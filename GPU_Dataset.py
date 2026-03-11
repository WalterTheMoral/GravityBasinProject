import cupy as cp
import numpy as np

# ---------- SIMULATION PARAMETERS ----------
BATCH_SIZE = 20000
NUM_BATCHES = 200

DT = 1/60
MAX_STEPS = 2000
EPSILON = 8.0

POINT_MASS = 50.0
ATTRACTOR_MASS = 1.0
G = 1e3
FRICTION = 0.01

MAX_DISTANCE = 2.0
MAX_VELOCITY = 1.0

OUTPUT_FILE = "basin_dataset_gpu.npz"
# ------------------------------------------


def random_points(n):
    return cp.random.uniform(0, 100, (n, 2))


def simulate_batch():

    # initial particle states
    pos = random_points(BATCH_SIZE)
    vel = cp.zeros((BATCH_SIZE, 2))

    # generate 3 attractors
    attractors = random_points(3)

    converged = cp.zeros(BATCH_SIZE, dtype=cp.bool_)
    basin = cp.full(BATCH_SIZE, -1, dtype=cp.int32)

    for _ in range(MAX_STEPS):

        # vector from particle -> attractor
        disp = attractors[None,:,:] - pos[:,None,:]

        dist = cp.linalg.norm(disp, axis=2)

        # gravitational magnitude
        force_mag = (G * POINT_MASS * ATTRACTOR_MASS) / (dist + EPSILON)**2

        direction = disp / (dist[:,:,None] + 1e-9)

        gravity = cp.sum(force_mag[:,:,None] * direction, axis=1)

        # friction
        friction = -FRICTION * vel

        force = gravity + friction

        # velocity update
        vel += (force / POINT_MASS) * DT

        # position update
        pos += vel * DT

        speed = cp.linalg.norm(vel, axis=1)

        # convergence check
        for i in range(3):

            d = cp.linalg.norm(pos - attractors[i], axis=1)

            mask = (
                (d < MAX_DISTANCE)
                & (speed < MAX_VELOCITY)
                & (~converged)
            )

            basin[mask] = i
            converged[mask] = True

        if cp.all(converged):
            break

    return pos, attractors, basin


def generate_dataset():

    features = []
    labels = []

    for batch in range(NUM_BATCHES):

        pos, attractors, basin = simulate_batch()

        pos = cp.asnumpy(pos)
        attractors = cp.asnumpy(attractors)
        basin = cp.asnumpy(basin)

        attractor_flat = attractors.flatten()

        attr = np.tile(attractor_flat, (pos.shape[0],1))

        X = np.hstack((pos, attr))
        y = basin

        features.append(X)
        labels.append(y)

        print("batch", batch)

    X = np.vstack(features).astype(np.float32)
    y = np.concatenate(labels).astype(np.int32)

    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=y
    )


if __name__ == "__main__":
    generate_dataset()