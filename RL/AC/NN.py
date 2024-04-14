import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def initialize_network(input_size, hidden_sizes, output_size):
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    weights = []
    biases = []
    np.random.seed(42)
    for i in range(len(layer_sizes) - 1):
        weights.append(np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * 0.01)
        biases.append(np.random.randn(layer_sizes[i + 1], 1) * 0.01)
    return weights, biases

def forward_propagate(X, weights, biases):
    X = X.reshape(-1, 1)
    activations = [X]
    for weight, bias in zip(weights, biases):
        net_input = np.dot(weight, activations[-1]) + bias
        activation = sigmoid(net_input)
        activations.append(activation)
    return activations

def backward_propagate(y, activations, weights):
    errors = [y - activations[-1]]
    deltas = [errors[-1] * sigmoid_derivative(activations[-1])]
    for i in reversed(range(len(weights) - 1)):
        delta = np.dot(weights[i + 1].T, deltas[0]) * sigmoid_derivative(activations[i + 1])
        deltas.insert(0, delta)
    return deltas

def update_weights(weights, biases, activations, deltas, eta=0.01):
    for i in range(len(weights)):
        weights[i] += eta * np.dot(deltas[i], activations[i].T)
        biases[i] += eta * deltas[i]

class Actor:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.weights, self.biases = initialize_network(input_size, hidden_sizes, output_size)
    
    def predict(self, state):
        activations = forward_propagate(state, self.weights, self.biases)
        return activations[-1]

class Critic:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.weights, self.biases = initialize_network(input_size, hidden_sizes, output_size)
    
    def evaluate(self, state, action):
        combined_input = np.concatenate((state, action), axis=0)
        activations = forward_propagate(combined_input, self.weights, self.biases)
        return activations[-1]
