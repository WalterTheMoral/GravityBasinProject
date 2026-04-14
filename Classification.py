import random

import pandas as pd
import numpy
import matplotlib.pyplot as plt
import time

from MachineLearningBase import *

# import DatabaseGeneration
import Simulation

import numpy as np

from Simulation import FixedMass, PointMass

database = np.load("basin_dataset_gpu_1E6_V2.npz")

X = database["X"].T
Y = num_to_one_hot(3, database["y"])

X = 2 * (X/100) - 1 # Normalise between [-1,1]

print(X.shape)
print(Y.shape)

database_examples = X.shape[1]
train_examples = database_examples * 9 // 10

X_Train = X[:, :train_examples]
X_Test = X[:, train_examples:]

Y_Train = Y[:, :train_examples]
Y_Test = Y[:, train_examples:]


# layers = (
#     LayerConfiguration("Input Layer", 32, (8,), Relu(), He(), Adam()),
#     LayerConfiguration("Hidden Layer 1", 32, (32,), Relu(), He(), Adam()),
#     LayerConfiguration("Hidden Layer 2", 32, (32,), Relu(), He(), Adam()),
#     LayerConfiguration("Softmax Layer", 3, (32,), Softmax(), Xaviar(), Adam()),
# )
layers = (
    LayerConfiguration("Input Layer", 32, (8,), Relu(), File("Saved Weights I2000/Layer0.h5"), Adam()),
    LayerConfiguration("Hidden Layer 1", 32, (32,), Relu(), File("Saved Weights I2000/Layer1.h5"), Adam()),
    LayerConfiguration("Hidden Layer 2", 32, (32,), Relu(), File("Saved Weights I2000/Layer2.h5"), Adam()),
    LayerConfiguration("Softmax Layer", 3, (32,), TrimSoftmax(), File("Saved Weights I2000/Layer3.h5"), Adam()),
)
model = Network(
    NetworkConfiguration("Model", CategoricalCrossEntropy())
)
model.add(*(Layer(config) for config in layers))

if __name__ == "__main__":
    print(X_Train.shape)
    print(Y_Train.shape)

    # start = time.time()
    # costs = model.train(X_Train, Y_Train, 2000)
    # print(time.time() - start)
    # model.save_weights("Saved Weights 2 I2000")
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations')
    # plt.title("Cost over time " + str(0.1))
    # plt.show()

    model.confusion_matrix(X_Train, Y_Train)
    model.confusion_matrix(X_Test, Y_Test)

    count, iter = 0, 300
    for i in range(iter):
        index = random.randint(0, X.shape[1] - 1)
        sample = []
        for _ in range(8): sample.append(random.random() * 100)
        sample = np.array([2 * x / 100 - 1 for x in sample])
        # print(*(((float(sample[i]), float(sample[i+1]))) for i in range(2, 7, 2)))
        sim = Simulation.Simulator([FixedMass((sample[i] + 1) * 50, (sample[i+1] + 1) * 50, 1) for i in range(2, 8, 2)],
                                   PointMass((sample[0] + 1) * 50 , (sample[1] + 1) * 50, 50, 1))
        if i % 10 == 0:
            print("Model", model.predict(sample.reshape(8,1)).T[0])
            print("Sim", num_to_one_hot(3, np.array([sim.converge_to_which_basin()])).T[0])
            print()
        # print(sample)
        # print(sample)
        # sample = X[:,index]
        # print(model.predict(sample.reshape(8,1)).T[0])
        # print(Y[:,index])
        # print(sample)
        equal = np.array_equal(model.predict(sample.reshape(8,1)).T[0], num_to_one_hot(3, np.array([sim.converge_to_which_basin()])).T[0])
        if equal: count += 1
        # print()
    print(count / iter)

    # prediction = np.argmax(model.predict(X_Train), axis=0)
    # sim = np.argmax(Y_Train, axis=0)
    # print(np.sum(prediction==sim))
    # print(prediction.shape)
    # print(np.sum(prediction==sim) / prediction.shape[0])

    # count = 0
    # for i in range(1, 1001):
    #     data = DatabaseGeneration.generate_sample(0)
    #     filtered_data = np.array(data[0:8]).T.reshape(-1, 1)
    #     prediction = np.argmax(model.predict(filtered_data), axis=0)[0]
    #     sim = data[8]
    #
    #     print(f"{sim} : {prediction}")
    #     count += sim == prediction
    #     print(f"{count} / {i}")

    # print(count)
