import numpy as np

np.random.seed(420)


class Perzeptron:

    def __init__(self, layers, learning_rates=None):
        if not 1 < len(layers) < 5 or any(layers) > 1000:
            print("Incorrect layer initialization")
            return

        self.network = list()

        self.weights, self.layer_count = [], len(layers) - 1    # input layer has no weights, not counted
        self.learning_rates = learning_rates if learning_rates else np.ones(self.layer_count-1)

        for i in range(self.layer_count):
            weights = np.random.uniform(-2, 2, (layers[i], layers[i+1]) )
            self.weights.append(weights)
            self.network.append({'weights': weights})

        self.bias = np.random.uniform(0, 1, (self.layer_count, 1))
        self.N, self.M = layers[0], layers[-1]



    def forward_propagate(self, inputs):
        # ACTIVATION FUNCTIONS DIFFERENT FOR HIDDEN LAYERS?
        for i in range(self.layer_count):
            inputs = np.dot(inputs, self.network[i]['weights']) + self.bias[i]
            inputs = np.tanh(inputs)    # activation
            self.network[i]['outputs'] = inputs
        return inputs


    def back_propagate(self, outputs, expected):
        for i in reversed(range(0, self.layer_count)):
            if i != self.layer_count - 1:
                outputs = self.network[i]['outputs'] 
                delta_below = self.network[i+1]['delta']
                delta = 1   # TODO: help how tf do i calculate this 
                self.network[i]['delta'] = delta
            else:   # runs first
                # output layer delta calc is at least correct
                errors = outputs - expected
                delta = errors * self.transfer_derivative(outputs)
                self.network[i]['delta'] = delta    # store delta for backprop

            self.network[i]['weights'] += self.learning_rates * delta * self.network[i]['outputs']  # TODO: fix, how?


    def transfer_derivative(self, outputs):
        # tanh derivative
        derivative = 1 - outputs ** 2
        return derivative


    def train(self, input_data, teacher_data):
        for inputs, labels in list(zip(input_data, teacher_data)):
            outputs = self.forward_propagate(inputs)
            error = outputs - labels
            self.back_propagate(outputs, labels)
