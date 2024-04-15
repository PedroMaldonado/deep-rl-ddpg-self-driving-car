import numpy as np
from OOPNNenhanced import NeuralNetwork

class ActorNetwork:
    def __init__(self, state_size, action_size, hidden_units, tau=0.001, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.lr = lr
        
        # Initialize primary and target networks with tanh activation for the output layer
        self.network = NeuralNetwork(state_size, hidden_units, action_size, activation='relu', output_activation='tanh')
        self.target_network = NeuralNetwork(state_size, hidden_units, action_size, activation='relu', output_activation='tanh')
        
        # Manually copy weights for target network initialization
        self.target_network.weights = [np.copy(w) for w in self.network.weights]
        self.target_network.biases = [np.copy(b) for b in self.network.biases]

    def train(self, states, action_gradients):
        # Perform forward pass
        activations = self.network.forward_propagate(states)
        
        # Convert action_gradients to deltas suitable for your MLP's backpropagation
        deltas = self.custom_backward(action_gradients, activations)
        
        # Update network weights
        self.network.update_weights(activations, deltas, self.lr)

    def custom_backward(self, action_gradients, activations):
        # Properly transform action gradients to match network architecture
        # Assume action_gradients are already negative as needed
        deltas = []
        for grad, activation in zip(action_gradients, reversed(activations[1:])):
            delta = grad * self.network.apply_activation_derivative(activation)
            deltas.insert(0, delta)
        return deltas

    def update_target_network(self):
        # Soft update target network's weights
        for idx, (main_weights, target_weights) in enumerate(zip(self.network.weights, self.target_network.weights)):
            target_weights = self.tau * main_weights + (1 - self.tau) * target_weights
            self.target_network.weights[idx] = target_weights

        for idx, (main_biases, target_biases) in enumerate(zip(self.network.biases, self.target_network.biases)):
            target_biases = self.tau * main_biases + (1 - self.tau) * target_biases
            self.target_network.biases[idx] = target_biases
