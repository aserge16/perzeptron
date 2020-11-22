import numpy as np

np.random.seed(420)


class Perzeptron:

    def __init__(self, layers):
        if not 1 < len(layers) < 5 or any(layers) > 1000:
            print("Incorrect layer initialization")
            return

        self.weights = []
        for i in range(len(layers) - 1):
            weights = np.random.uniform(-2, 2, (layers[i], layers[i+1]) )
            self.weights.append(weights)
        self.bias = np.random.uniform(0, 1, (len(layers), 1))
        self.hidden_count = len(layers) - 2
        self.N, self.M = layers[0], layers[-1]


    def predict(self, inputs):
        # ACTIVATION FUNCTIONS DIFFERENT FOR HIDDEN LAYERS?
        for i in range(self.hidden_count):
            inputs = np.dot(inputs, self.weights[i]) + self.bias[i]
            inputs = np.tanh(inputs)    # activation

        # Output, here can apply different activation function
        inputs = np.dot(inputs, self.weights[-1]) + self.bias[-1]
        inputs = np.tanh(inputs)
        return inputs


    def train(self, input_data, teacher_data):
        for inputs, labels in list(zip(input_data, teacher_data)):
            outputs = self.predict(inputs)
            final_error = outputs - labels
