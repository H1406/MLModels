import LinearRegressionModel as lrm
import NeuralNetwork as nnm
import numpy as np


X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])  # XOR problem

nn = nnm.NeuralNetwork(layers=[2,3,1])  # Flexible architecture
nn.fit(X, y)
predictions = nn.predict(X)

print("OOP Predictions:", predictions)


