import pandas as pd
import numpy
import matplotlib.pyplot as plt
import time

from MachineLearningBase import *

import DatabaseGeneration
import Simulation

import numpy as np

# database = np.load("database_v2.npy")
# print(database.shape)
# print(database.dtype)
# print(database[:5])
#
# np.random.shuffle(database)
# X = database[:, :-2].T
# Y = num_to_one_hot(3, database[:,8])
#
# X = 2 * (X/100) - 1 # Normalise between [-1,1]
#
# print(X.shape)
# print(Y.shape)
#
# database_examples = X.shape[1]
# train_examples = database_examples * 9 // 10
#
# X_Train = X[:, :train_examples]
# X_Test = X[:, train_examples:]
#
# Y_Train = Y[:, :train_examples]
# Y_Test = Y[:, train_examples:]


layers = (
    LayerConfiguration("Input Layer", 32, (8,), Relu(), He(), Adam()),
    LayerConfiguration("Hidden Layer 1", 32, (32,), Relu(), He(), Adam()),
    LayerConfiguration("Softmax Layer", 3, (32,), Softmax(), Xaviar(), Adam()),
)
# layers = (
#     LayerConfiguration("Input Layer", 32, (8,), Relu(), File("Saved Weights/Layer0.h5"), Adam()),
#     LayerConfiguration("Hidden Layer 1", 32, (32,), Relu(), File("Saved Weights/Layer1.h5"), Adam()),
#     LayerConfiguration("Softmax Layer", 3, (32,), TrimSoftmax(), File("Saved Weights/Layer2.h5"), Adam()),
# )
model = Network(
    NetworkConfiguration("Model", CategoricalCrossEntropy())
)
model.add(*(Layer(config) for config in layers))

if __name__ == "__main__":
    print(X_Train.shape)
    print(Y_Train.shape)

    start = time.time()
    costs = model.train(X_Train, Y_Train, 2000)
    print(time.time() - start)
    model.save_weights("Saved Weights")
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Cost over time " + str(0.1))
    plt.show()

    model.confusion_matrix(X_Train, Y_Train)
    model.confusion_matrix(X_Test, Y_Test)

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
