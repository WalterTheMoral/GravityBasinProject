import pandas as pd
import numpy
import matplotlib.pyplot as plt
import time

from MachineLearningBase import *

database = pd.read_csv("fixed_attractors.csv")
X = database.to_numpy()[:, :2].T
Y = num_to_one_hot(3, database.to_numpy()[:, 8])

X = 2 * (X/100) - 1 # Normalise between [-1,1]

print(X.shape)
print(Y.shape)

database_examples = X.shape[1]
train_examples = database_examples * 9 // 10

X_Train = X[:, :train_examples]
X_Test = X[:, train_examples:]

Y_Train = Y[:, :train_examples]
Y_Test = Y[:, train_examples:]


layers = (
    LayerConfiguration("Input Layer", 32, (2,), Relu(), He(), Adam()),
    LayerConfiguration("Hidden Layer 1", 32, (32,), Relu(), He(), Adam()),
    LayerConfiguration("Softmax Layer", 3, (32,), Softmax(), Xaviar(), Adam()),
)
model = Network(
    NetworkConfiguration("Model", CategoricalCrossEntropy())
)
model.add(*(Layer(config) for config in layers))

print(X_Train.shape)
print(Y_Train.shape)

start = time.time()
costs = model.train(X_Train, Y_Train, 1000)
print(time.time() - start)
model.save_weights("Digits")
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Cost over time " + str(0.1))
plt.show()

model.confusion_matrix(X_Train, Y_Train)
model.confusion_matrix(X_Test, Y_Test)
