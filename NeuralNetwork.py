import numpy as np
class NeuralNetwork:

    def __init__(self, layers, lr=0.1, epochs=10000):
        self.layers = layers
        self.lr = lr
        self.epochs = epochs
        # Initialize weights and biases randomly

        self.weights = [np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2.0 / self.layers[i])
                        for i in range(len(self.layers) - 1)]
        self.biases = [np.zeros((1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    def fit(self, X, y):
        for _ in range(self.epochs):
            # Forward Pass
            activations = [X]
            for i in range(len(self.weights)):
                z = np.dot(activations[-1],self.weights[i]) + self.biases[i]
                activations.append(self.sigmoid(z))

            #backward propagation
            error = y - activations[-1]
            deltas = [error * self.sigmoid_derivative(activations[-1])]
            for i in range(len(self.weights) - 1, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[i].T) * self.sigmoid_derivative(activations[i]))
            deltas.reverse()

            # Update weights and biases
            for i in range(len(self.weights)):
                self.weights[i] += activations[i].T.dot(deltas[i]) * self.lr
                self.biases[i] += np.sum(deltas[i], axis=0, keepdims= True) * self.lr

    def predict(self, X):
        output = X.copy()  # Avoid modifying input
        for i in range(len(self.weights)):
            output = self.sigmoid(np.dot(output, self.weights[i]) + self.biases[i])
        return output