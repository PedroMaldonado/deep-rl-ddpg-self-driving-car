import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        # Initialize network dimensions
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        self.activation = activation  # Choose between 'relu' and 'sigmoid' for hidden layers

        # Seed for reproducibility
        np.random.seed(42)

        # Initialize weights and biases
        for i in range(len(self.layer_sizes) - 1):
            if self.activation == 'relu':
                # He initialization suitable for ReLU
                std = np.sqrt(2.0 / self.layer_sizes[i])
            else:
                # Standard small number initialization for sigmoid
                std = 0.01
            # Weights are initialized with a normal distribution scaled by `std`
            self.weights.append(np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i]) * std)
            # Biases are initialized as zeros to start with no influence
            self.biases.append(np.zeros((self.layer_sizes[i + 1], 1)))

    @staticmethod
    def sigmoid(x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(output):
        # Derivative of sigmoid, output is the sigmoid function result
        return output * (1 - output)

    @staticmethod
    def relu(x):
        # ReLU activation function
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        # Derivative of ReLU, 1 for x > 0 otherwise 0
        return (x > 0).astype(float)

    def forward_propagate(self, X):
        # Reshape input to ensure it is a column vector
        X = X.reshape(-1, 1)
        activations = [X]  # Start with input activation

        # Forward pass through each layer
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            net_input = np.dot(weight, activations[-1]) + bias  # Compute Wx + b
            if index < len(self.weights) - 1 and self.activation == 'relu':
                activation = self.relu(net_input)  # ReLU for hidden layers
            else:
                activation = self.sigmoid(net_input)  # Sigmoid for output layer
            activations.append(activation)
        return activations

    def backward_propagate(self, y, activations):
        # Compute the error at the output
        errors = [y - activations[-1]]
        # Apply the derivative of sigmoid since output layer uses sigmoid
        deltas = [errors[-1] * self.sigmoid_derivative(activations[-1])]

        # Backpropagation through layers, updating deltas
        for i in reversed(range(len(self.weights) - 1)):
            if self.activation == 'relu':
                # Calculate deltas using ReLU derivative
                delta = np.dot(self.weights[i + 1].T, deltas[0]) * self.relu_derivative(activations[i + 1])
            else:
                # Calculate deltas using sigmoid derivative for consistency
                delta = np.dot(self.weights[i + 1].T, deltas[0]) * self.sigmoid_derivative(activations[i + 1])
            deltas.insert(0, delta)
        return deltas

    def update_weights(self, activations, deltas, eta=0.01):
        # Update weights and biases based on deltas and activations
        for i in range(len(self.weights)):
            self.weights[i] += eta * np.dot(deltas[i], activations[i].T)  # Adjust weights by learning rate and delta
            self.biases[i] += eta * deltas[i]  # Adjust biases similarly

# Usage example:
# nn = NeuralNetwork(input_size=5, hidden_sizes=[10, 10], output_size=1, activation='relu')
