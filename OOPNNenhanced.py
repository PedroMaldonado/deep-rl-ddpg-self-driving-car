import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', output_activation=None):
        output_activation = output_activation if output_activation else activation
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.activation = activation  # 'relu' for hidden layers, potentially 'tanh' for the output layer
        self.weights = []
        self.biases = []

        np.random.seed(42)  # Seed for reproducibility

        # for i in range(len(self.layer_sizes) - 1):
        #     # He initialization for ReLU or Glorot initialization for tanh and sigmoid
        #     if self.activation == 'relu' and i < len(self.layer_sizes) - 2:
        #         std = np.sqrt(2 / self.layer_sizes[i])  # He initialization
        #     else:
        #         limit = np.sqrt(6 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))  # Glorot initialization
        #         self.weights.append(np.random.uniform(-limit, limit, (self.layer_sizes[i + 1], self.layer_sizes[i])))
        #     self.biases.append(np.zeros((self.layer_sizes[i + 1], 1)))

        # for i in range(len(self.layer_sizes)-1):
        #     if i == len(self.layer_sizes) - 2:  # Last layer uses output activation
        #         std = np.sqrt(2. / (self.layer_sizes[i] + self.layer_sizes[i+1])) if output_activation == 'relu' else np.sqrt(1. / (self.layer_sizes[i] + self.layer_sizes[i+1]))
        #     else:
        #         std = np.sqrt(2. / (self.layer_sizes[i] + self.layer_sizes[i+1]))
        #     self.weights.append(std * np.random.randn(self.layer_sizes[i+1], self.layer_sizes[i]))
        #     self.biases.append(np.zeros((self.layer_sizes[i+1], 1)))

        for i in range(len(self.layer_sizes) - 1):
            self.weights.append(np.random.randn(self.layer_sizes[i+1], self.layer_sizes[i]) * np.sqrt(2. / self.layer_sizes[i]))
            self.biases.append(np.zeros((self.layer_sizes[i+1], 1)))

    def forward_propagate(self, X):
        X = X.reshape(self.layer_sizes[0], -1)  # Ensure input is a column vector
        activations = [X]
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):

            print("Weight shape:", weight.shape, "Activation shape:", activations[-1].shape)
            z = np.dot(weight, activations[-1]) + bias
            a = self.apply_activation(z, index)
            activations.append(a)
        return activations[-1]

    # def apply_activation(self, z, index):
    #     if index < len(self.weights) - 1:
    #         if self.activation == 'relu':
    #             return np.maximum(0, z)
    #     return np.tanh(z)

    def apply_activation(self, z, index):
        if index < len(self.weights) - 1:
            return np.maximum(0, z)  # ReLU
        else:
            return np.tanh(z)

    def backward_propagate(self, y, activations):
        errors = [y - activations[-1]]
        deltas = [errors[-1] * self.apply_activation_derivative(activations[-1])]

        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(self.weights[i + 1].T, deltas[0]) * self.apply_activation_derivative(activations[i + 1])
            deltas.insert(0, delta)
        return deltas

    def apply_activation_derivative(self, output):
        if self.activation == 'relu':
            return (output > 0).astype(float)
        return 1 - np.tanh(output)**2

    def update_weights(self, activations, deltas, eta=0.01):
        for i in range(len(self.weights)):
            dw = np.dot(deltas[i], activations[i].T)
            db = deltas[i]
            self.weights[i] += eta * dw
            self.biases[i] += eta * db.mean(axis=1, keepdims=True)

# Example usage:
# nn = NeuralNetwork(input_size=5, hidden_sizes=[10, 10], output_size=1, activation='relu')
