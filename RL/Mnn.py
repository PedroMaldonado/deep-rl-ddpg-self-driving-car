import numpy as np
import pandas as pd
import pickle
import time

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(next_layer, current_layer) for current_layer, next_layer in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = np.random.randn(len(layer_sizes)-1)
        self.activations = []
        self.z_values = []
        
    def forward_propagate(self, X):
        if len(X) != self.layer_sizes[0]: 
            raise Exception("Wrong number of inputs")
        
        # Forward propagation
        activation = X
        self.activations = [np.array(X)]
        self.z_values = []
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            activation = self.sigmoid(z)
            
            self.activations.append(np.array(activation))
            self.z_values.append(np.array(z))
        
        return activation
    
    def backpropagate(self, targets, learning_rate):
        if len(targets) != self.layer_sizes[-1]: 
            raise Exception("Wrong number of targets")
        
        # Calculate the output layer error
        output_error = self.activations[-1] - targets
        layer_error = output_error
        
        # Loop through layers backwards starting from the output layer to the first hidden layer
        for i in reversed(range(len(self.layer_sizes) - 1)):
            # Calculate gradient for weights and biases
            weight_gradient = np.dot(layer_error.reshape(-1, 1), self.activations[i].reshape(1, -1))
            bias_gradient = layer_error
            
            # Update weights and biases
            self.weights[i] -= learning_rate * weight_gradient
            self.biases[i] -= learning_rate * np.sum(bias_gradient)
            
            # Calculate error for the next layer
            if i > 0:
                layer_error = np.dot(self.weights[i].T, layer_error) * self.sigmoid_derivative(self.z_values[i-1])
        
    def saveweights(self, filename):
        pickle.dump((self.layer_sizes, self.weights, self.biases), open(filename, "wb"))
        
    def loadweights(self, filename):
        self.layer_sizes, self.weights, self.biases = pickle.load(open(filename, "rb"))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

epochs = 1000

from ucimlrepo import fetch_ucirepo 

iris = fetch_ucirepo(id=53) 

X = iris.data.features.apply(lambda x: (x - x.mean()) / x.std())
y = pd.get_dummies(iris.data.targets)

# layer_sizes = [input, hidden1, hidden2, ..., hiddenN, output]
# layer_sizes = [len(X.columns), 25, 50, 100, 50, 25, len(y.iloc[0])]
layer_sizes = [len(X.columns), 20, 20, 20, len(y.iloc[0])]
nn = NeuralNetwork(layer_sizes)

start_time = time.process_time()

for i in range(epochs):
    selection = np.random.choice(len(X), len(X), replace=False)
    for x in range(len(X)):
        case = X.iloc[selection[x]].values
        expected = y.iloc[selection[x]].values
        prediction = nn.forward_propagate(case)
        nn.backpropagate(expected, 0.2)

    quadratic_error = []
    # Test all cases in the dataset
    for x in range(len(X)):
        case = X.iloc[x].values
        expected = y.iloc[x].values
        prediction = nn.forward_propagate(case)
        error = np.sum(np.square(expected - prediction))
        quadratic_error.append(error)
        
    print(np.mean(quadratic_error))
    if (np.mean(quadratic_error) < 0.25):
        break
    
print("Time: " + str(time.process_time() - start_time))