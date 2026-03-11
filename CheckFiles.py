from MachineLearningBase import *

# layer = Layer(LayerConfiguration("Layer", 5, (1,), Relu(), Random()))
layer = Layer(LayerConfiguration("Layer", 5, (1,), Relu(), File("Check/Layer0.h5")))
model = Network(NetworkConfiguration("Model", CategoricalCrossEntropy()))
model.add(layer)
model.save_weights("Check")

print(model.layers[0].W)
print(model.layers[0].W.shape)
print(model.layers[0].b)
print(model.layers[0].b.shape)
