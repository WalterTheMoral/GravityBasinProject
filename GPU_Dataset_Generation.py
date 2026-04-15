import cupy as cp
import numpy as np
import time

# ---------- SIMULATION PARAMETERS ----------
BATCH_SIZE = 50000
NUM_BATCHES = 20

DT = 1/60
MAX_STEPS = 200000
EPSILON = 8.0

POINT_MASS = 50.0
ATTRACTOR_MASS = 1.0
G = 1
FRICTION = 0.05

MAX_DISTANCE = 3.0
MAX_VELOCITY = 3.0

OUTPUT_FILE = "basin_dataset_gpu_1E6_V2.npz"
# ------------------------------------------


def random_points(n):
    return cp.random.uniform(0, 100, (n, 2))


def simulate_batch():

    # particle positions
    pos = random_points(BATCH_SIZE)
    initial_pos = pos.copy()

    # velocities
    vel = cp.zeros((BATCH_SIZE, 2))

    # each particle gets its own 3 attractors
    attractors = cp.random.uniform(0, 100, (BATCH_SIZE, 3, 2))

    converged = cp.zeros(BATCH_SIZE, dtype=cp.bool_)
    basin = cp.full(BATCH_SIZE, -1, dtype=cp.int32)

    start = time.time()
    for step in range(MAX_STEPS):
        if step % 1000 == 0:
            cp.cuda.Stream.null.synchronize()

            converged_count = int(cp.sum(converged).get())

            print(
                f"step {step} | "
                f"time {time.time()-start:.2f}s | "
                f"converged {converged_count}/{BATCH_SIZE}"
            )

            start = time.time()

        active = ~converged
        if not cp.any(active):
            break

        pos_a = pos[active]
        vel_a = vel[active]
        attr_a = attractors[active]

        # particle -> attractor displacement
        disp = attr_a - pos_a[:, None, :]

        dist = cp.linalg.norm(disp, axis=2)

        force_mag = (G * POINT_MASS * ATTRACTOR_MASS) / (dist + EPSILON)**2

        direction = disp / (dist[:, :, None] + 1e-9)

        gravity = cp.sum(force_mag[:, :, None] * direction, axis=1)

        friction = -FRICTION * vel_a

        force = gravity + friction

        vel_a += (force / POINT_MASS) * DT
        pos_a += vel_a * DT

        vel[active] = vel_a
        pos[active] = pos_a

        speed = cp.linalg.norm(vel_a, axis=1)

        for i in range(3):

            d = cp.linalg.norm(pos_a - attr_a[:, i, :], axis=1)

            mask = (
                (d < MAX_DISTANCE)
                & (speed < MAX_VELOCITY)
            )

            idx = cp.where(active)[0][mask]

            basin[idx] = i
            converged[idx] = True

    return initial_pos, attractors, basin


def generate_dataset():

    features = []
    labels = []

    for batch in range(NUM_BATCHES):

        pos, attractors, basin = simulate_batch()

        pos = cp.asnumpy(pos)
        attractors = cp.asnumpy(attractors)
        basin = cp.asnumpy(basin)

        attr_flat = attractors.reshape(BATCH_SIZE, 6)

        X = np.hstack((pos, attr_flat))
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
    import cupy as cp
    print(cp.cuda.runtime.getDeviceProperties(0)["name"])

    import time

    x = cp.random.rand(10_000_000)

    start = time.time()
    y = cp.sqrt(x)
    cp.cuda.Stream.null.synchronize()

    print("time:", time.time() - start)

    print("Start")
    generate_dataset()

    import numpy as np

    data = np.load(OUTPUT_FILE)

    X = data["X"]
    y = data["y"]

    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)

    print("\nFirst 10 feature rows:")
    print(X[:10])

    print("\nFirst 10 labels:")
    print(y[:10])

    print("Unconverged:", (y == -1).sum())
    print("Total:", len(y))