import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        np.random.seed(42)
        for i in range(len(self.layer_sizes) - 1):
            self.weights.append(np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i]) * 0.01)
            self.biases.append(np.random.randn(self.layer_sizes[i + 1], 1) * 0.01)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return NeuralNetwork.sigmoid(x) * (1 - NeuralNetwork.sigmoid(x))

    def forward_propagate(self, X):
        X = X.reshape(-1, 1)
        activations = [X]
        for weight, bias in zip(self.weights, self.biases):
            net_input = np.dot(weight, activations[-1]) + bias
            activation = self.sigmoid(net_input)
            activations.append(activation)
        return activations

    def backward_propagate(self, y, activations):
        errors = [y - activations[-1]]
        deltas = [errors[-1] * self.sigmoid_derivative(activations[-1])]
        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(self.weights[i + 1].T, deltas[0]) * self.sigmoid_derivative(activations[i + 1])
            deltas.insert(0, delta)
        return deltas

    def update_weights(self, activations, deltas, eta=0.01):
        for i in range(len(self.weights)):
            self.weights[i] += eta * np.dot(deltas[i], activations[i].T)
            self.biases[i] += eta * deltas[i]