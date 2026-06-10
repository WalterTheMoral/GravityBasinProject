import matplotlib.pyplot as plt
from MachineLearningBase import *
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
#     LayerConfiguration("Input Layer", 32, (8,), Relu(), He(), Adam(), L2()),
#     LayerConfiguration("Hidden Layer 1", 32, (32,), Relu(), He(), Adam(), L2()),
#     LayerConfiguration("Hidden Layer 2", 32, (32,), Relu(), He(), Adam()),
#     LayerConfiguration("Softmax Layer", 3, (32,), TrimSoftmax(), Xaviar(), Adam(), L2()),
# )
layers = (
    LayerConfiguration("Input Layer", 32, (8,), Relu(), File("Saved Weights I3_000 (32,32,32,3)/Layer0.h5"), Adam(), L2()),
    LayerConfiguration("Hidden Layer 1", 32, (32,), Relu(), File("Saved Weights I3_000 (32,32,32,3)/Layer1.h5"), Adam(), L2()),
    LayerConfiguration("Hidden Layer 2", 32, (32,), Relu(), File("Saved Weights I3_000 (32,32,32,3)/Layer2.h5"), Adam(), L2()),
    LayerConfiguration("Softmax Layer", 3, (32,), TrimSoftmax(), File("Saved Weights I3_000 (32,32,32,3)/Layer3.h5"), Adam(), L2()),
)
model = Network(
    NetworkConfiguration("Model", CategoricalCrossEntropy())
)
model.add(*(Layer(config) for config in layers))

if __name__ == "__main__":
    print(X_Train.shape)
    print(Y_Train.shape)

    # start = time.time()
    # costs = model.train(X_Train, Y_Train, 3_000)
    # print(time.time() - start)
    # model.save_weights("Saved Weights I3_000 (32,32,32,3)")
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations')
    # plt.title("Cost over time " + str(0.1))
    # plt.show()

    train_matrix = model.confusion_matrix(X_Train, Y_Train)
    test_matrix = model.confusion_matrix(X_Test, Y_Test)

    train_matrix /= np.sum(train_matrix)
    print(train_matrix.round(3))

    test_matrix /= np.sum(test_matrix)
    print(test_matrix.round(3))
