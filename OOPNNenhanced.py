import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.activation = activation  # Choose between 'relu' and 'sigmoid' for hidden layers
        self.weights = []
        self.biases = []

        np.random.seed(42)  # Seed for reproducibility

        # Initialize weights and biases with improved initialization
        for i in range(len(self.layer_sizes) - 1):
            # He initialization for ReLU or Glorot initialization for others
            if self.activation == 'relu':
                std = np.sqrt(2 / self.layer_sizes[i])  # He initialization
            else:
                limit = np.sqrt(6 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))  # Glorot initialization
                self.weights.append(np.random.uniform(-limit, limit, (self.layer_sizes[i + 1], self.layer_sizes[i])))
            self.biases.append(np.zeros((self.layer_sizes[i + 1], 1)))

    def forward_propagate(self, X):
        X = X.reshape(-1, 1)  # Ensure input is a column vector
        activations = [X]
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(weight, activations[-1]) + bias
            a = self.apply_activation(z, index)
            activations.append(a)
        return activations

    def apply_activation(self, z, index):
        # Use ReLU for hidden layers and sigmoid for the output layer
        if index < len(self.weights) - 1 and self.activation == 'relu':
            return np.maximum(0, z)
        else:
            return 1 / (1 + np.exp(-z))

    def backward_propagate(self, y, activations):
        # Calculate the output layer error and its delta
        errors = [y - activations[-1]]
        deltas = [errors[-1] * self.apply_activation_derivative(activations[-1])]

        # Propagate errors backwards through the network
        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(self.weights[i + 1].T, deltas[0]) * self.apply_activation_derivative(activations[i + 1])
            deltas.insert(0, delta)
        return deltas

    def apply_activation_derivative(self, output):
        # Derivative of the activation function used in the backward propagation
        if self.activation == 'relu':
            return (output > 0).astype(float)
        else:
            return output * (1 - output)

    def update_weights(self, activations, deltas, eta=0.01):
        # Update weights and biases using simple gradient descent
        for i in range(len(self.weights)):
            dw = np.dot(deltas[i], activations[i].T)
            db = deltas[i]
            self.weights[i] += eta * dw
            self.biases[i] += eta * db.mean(axis=1, keepdims=True)

# Example of usage:
# nn = NeuralNetwork(input_size=5, hidden_sizes=[10, 10], output_size=1, activation='relu')
