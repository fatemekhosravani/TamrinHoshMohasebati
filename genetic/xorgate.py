import numpy as np

class NeuralNetwork:
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights_0 = 2 * np.random.random((2, 3)) - 1
        self.synaptic_weights_1 = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, iterations):
        for iteration in range(iterations):
            layer_0 = training_inputs
            layer_1 = self.sigmoid(np.dot(layer_0, self.synaptic_weights_0))
            layer_2 = self.sigmoid(np.dot(layer_1, self.synaptic_weights_1))
            layer_2_error = training_outputs - layer_2
            layer_2_delta = layer_2_error * self.sigmoid_derivative(layer_2)
            layer_1_error = layer_2_delta.dot(self.synaptic_weights_1.T)
            layer_1_delta = layer_1_error * self.sigmoid_derivative(layer_1)
            self.synaptic_weights_1 += layer_1.T.dot(layer_2_delta)
            self.synaptic_weights_0 += layer_0.T.dot(layer_1_delta)

    def predict(self, inputs):
        layer_1 = self.sigmoid(np.dot(inputs, self.synaptic_weights_0))
        layer_2 = self.sigmoid(np.dot(layer_1, self.synaptic_weights_1))
        return layer_2
training_inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
training_outputs = np.array([[0], [1], [1], [0]])
neural_network = NeuralNetwork()
neural_network.train(training_inputs, training_outputs, iterations=60000)
print("Predictions after training:")
for i in range(len(training_inputs)):
    print("Input:", training_inputs[i], "Output:", neural_network.predict(training_inputs[i]))
