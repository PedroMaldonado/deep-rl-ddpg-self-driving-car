import numpy as np
from OOPNNenhanced import NeuralNetwork
class CriticNetwork:
    def __init__(self, state_size, action_size, hidden_units, tau=0.001, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.lr = lr
        
        # Initialize primary and target networks
        self.network = NeuralNetwork(state_size + action_size, hidden_units, 1)
        self.target_network = NeuralNetwork(state_size + action_size, hidden_units, 1)

        # Manually copy weights for target network initialization
        self.target_network.weights = [np.copy(w) for w in self.network.weights]
        self.target_network.biases = [np.copy(b) for b in self.network.biases]
        # self.target_network.weights = self.network.weights.copy()
        # self.target_network.biases = self.network.biases.copy()

    def train(self, state, action, target):
        # # Forward propagate to get current prediction
        # # combined_input = np.concatenate([state, action])
        print("Shape of states in critic.train: ", state.shape)
        print("Shape of actions in critic.train: ", action.shape)

        state = np.atleast_2d(state)
        action = np.atleast_2d(action)

        # # Ensure they're properly shaped for concatenation
        if state.shape[0] != action.shape[0]:
            # Reshape actions if necessary (common adjustment for consistency)

            # action = action.reshape(state.shape[0], -1)
            raise ValueError("Mismatched batch sizes: states and actions must have the same number of samples")

        combined_input = np.concatenate([state, action], axis=1)  # Concatenate along axis 1


        activations = self.network.forward_propagate(combined_input)
        # Calculate gradients and update weights
        deltas = self.network.backward_propagate(target, activations)
        self.network.update_weights(activations, deltas, self.lr)

    def update_target_network(self):
        # Soft update target network's weights
        for idx, (main_weights, target_weights) in enumerate(zip(self.network.weights, self.target_network.weights)):
            target_weights = self.tau * main_weights + (1 - self.tau) * target_weights
            self.target_network.weights[idx] = target_weights

        for idx, (main_biases, target_biases) in enumerate(zip(self.network.biases, self.target_network.biases)):
            target_biases = self.tau * main_biases + (1 - self.tau) * target_biases
            self.target_network.biases[idx] = target_biases

    def compute_loss(predictions, targets):
        return np.mean((np.array(predictions) - np.array(targets)) ** 2)
