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
        # X = X.reshape(-1, 1)
        # X = X.reshape(-1, self.layer_sizes[0])
        if X.shape[0] != self.layer_sizes[0]:
            X = X.reshape(self.layer_sizes[0], -1)

        activations = [X]
        for weight, bias in zip(self.weights, self.biases):
            print("Weight shape:", weight.shape, "Activation shape:", activations[-1].shape)
            net_input = np.dot(weight, activations[-1]) + bias
            activation = self.sigmoid(net_input)
            activations.append(activation)
        return activations[-1]

    # def backward_propagate(self, y, activations):
    #     print("Number of weight matrices:", len(self.weights))
    #     print("Number of activations:", len(activations))

    #     errors = [y - activations[-1]]
    #     deltas = [errors[-1] * self.sigmoid_derivative(activations[-1])]

    #     print("Starting backpropagation...")
    #     print("Total activations recorded:", len(activations))
    #     print("Number of weight matrices:", len(self.weights))
    #     print("Number of activations:", len(activations))
    #     for i in reversed(range(len(self.weights) - 1)):
    #         print("Backpropagating from layer {} to layer {}".format(i+2, i+1))
    #         # print("Episode: {}, Total Reward: {}".format(i, total_reward))
    #         print("weights[{}].T shape: {}, deltas[0] shape: {}".format(i+1,self.weights[i+1].T.shape, deltas[0].shape))
            

    #         # delta = np.dot(self.weights[i + 1].T, deltas[0]) * self.sigmoid_derivative(activations[i])
    #         # deltas.insert(0, delta)

    #         if i < len(activations) - 1:
    #             delta = np.dot(self.weights[i+1].T, deltas[0]) * self.sigmoid_derivative(activations[i])
    #             deltas.insert(0, delta)
    #             print("activations[{}] shape: {}".format(i, activations[i].shape))
    #         else:
    #             print("Skipping activation index {} as it is out of bounds.".format(i))
    #     return deltas

    # def backward_propagate(self, y, activations):
    #     print("Number of weight matrices:", len(self.weights))
    #     print("Number of activations:", len(activations))

    #     if y.shape != activations[-1].shape:
    #         raise ValueError("Shape mismatch between output and target: {} vs {}".format(activations[-1].shape, y.shape))

    #     errors = [y - activations[-1]]
    #     deltas = [errors[-1] * self.sigmoid_derivative(activations[-1])]

    #     print("Starting backpropagation...")
    #     for i in reversed(range(len(self.weights) - 1)):
    #         print("Backpropagating from layer {} to layer {}".format(i+2, i+1))
    #         print("weights[{}].T shape: {}, deltas[0] shape: {}".format(i+1, self.weights[i+1].T.shape, deltas[0].shape))

    #         if i < len(activations) - 1:
    #             delta = np.dot(self.weights[i+1].T, deltas[0]) * self.sigmoid_derivative(activations[i])
    #             deltas.insert(0, delta)
    #             print("activations[{}] shape: {}".format(i, activations[i].shape))
    #         else:
    #             print("Skipping activation index {} as it is out of bounds.".format(i))

    #     return deltas

    def backward_propagate(self, y, activations):
        print("Number of weight matrices:", len(self.weights))
        print("Number of activations:", len(activations))

        # Explicitly reshape the last activation and re-assign it to ensure changes are applied
        if activations[-1].ndim == 1 or activations[-1].shape[0] != 1:
            activations[-1] = activations[-1].reshape(1, -1)
            print("Corrected activations[-1] shape to (1, N):", activations[-1].shape)

        # Ensure y is correctly shaped
        if y.ndim == 1 or y.shape[0] != 1:
            y = y.reshape(1, -1)
            print("Corrected target 'y' shape to (1, N):", y.shape)

        # Verify that shapes now match
        if activations[-1].shape != y.shape:
            raise ValueError("Shape mismatch between output and target after reshaping: {} vs {}".format(activations[-1].shape, y.shape))

        errors = [y - activations[-1]]
        deltas = [errors[-1] * self.sigmoid_derivative(activations[-1])]

        print("Starting backpropagation...")
        for i in reversed(range(len(self.weights) - 1)):
            print("Backpropagating from layer {} to layer {}".format(i+2, i+1))
            print("weights[{}].T shape: {}, deltas[0] shape: {}".format(i+1, self.weights[i+1].T.shape, deltas[0].shape))

            if i < len(activations) - 1:
                delta = np.dot(self.weights[i+1].T, deltas[0]) * self.sigmoid_derivative(activations[i])
                deltas.insert(0, delta)
                print("activations[{}] shape: {}".format(i, activations[i].shape))
            else:
                print("Skipping activation index {} as it is out of bounds.")

        return deltas




    # def update_weights(self, activations, deltas, eta=0.01):
    #     for i in range(len(self.weights)):
    #         self.weights[i] += eta * np.dot(deltas[i], activations[i].T)
    #         self.biases[i] += eta * deltas[i]

    # def update_weights(self, activations, deltas, eta=0.01):
    # # Assuming deltas are calculated per batch, average over the batch dimension
    #     for i in range(len(self.weights)):
    #         # Update weights
    #         self.weights[i] += eta * np.dot(deltas[i], activations[i].T)
            
    #         # Update biases, average deltas over the batch dimension if necessary
    #         bias_shape = self.biases[i].shape
    #         if deltas[i].ndim > 1:  # Check if deltas have a batch dimension
    #             mean_deltas = np.mean(deltas[i], axis=1, keepdims=True)  # Average over the batch
    #         else:
    #             mean_deltas = deltas[i]
            
    #         # Ensure bias shapes and delta shapes align
    #         if self.biases[i].shape != mean_deltas.shape:
    #             raise ValueError("Shape mismatch in biases and deltas: {} vs {}".format(self.biases[i].shape, mean_deltas.shape))
            
    #         self.biases[i] += eta * mean_deltas

    # def update_weights(self, activations, deltas, eta=0.01):
    #     for i in range(len(self.weights)):
    #         # Update weights
    #         self.weights[i] += eta * np.dot(deltas[i], activations[i].T)
            
    #         # Update biases, ensure the deltas are correctly reshaped if necessary
    #         bias_shape = self.biases[i].shape
    #         if deltas[i].shape[1] != 1:  # If deltas are not already in the shape (N, 1) where N is the number of neurons
    #             # Sum or average deltas along the batch dimension
    #             delta_for_biases = np.sum(deltas[i], axis=1, keepdims=True)
    #         else:
    #             delta_for_biases = deltas[i]
            
    #         if delta_for_biases.shape != bias_shape:
    #             raise ValueError("Expected delta shape {}, but got {}".format(bias_shape, delta_for_biases.shape))
            
    #         self.biases[i] += eta * delta_for_biases

    def update_weights(self, activations, deltas, eta=0.01):
        for i in range(len(self.weights)):
            # Update weights normally
            self.weights[i] += eta * np.dot(deltas[i], activations[i].T)
            
            # Update biases - ensure deltas are summed along the batch dimension
            if deltas[i].ndim > 1:  # More than one dimension, sum along the batch dimension
                delta_for_biases = np.sum(deltas[i], axis=0, keepdims=True)  # Sum along the batch (first) dimension
            else:
                delta_for_biases = deltas[i]
            
            # Ensure the delta_for_biases is correctly shaped for the biases
            if delta_for_biases.shape != self.biases[i].shape:
                # Proper reshaping to match the biases' shape
                delta_for_biases = delta_for_biases.reshape(self.biases[i].shape)
            
            # Update the biases
            self.biases[i] += eta * delta_for_biases


