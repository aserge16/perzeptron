import numpy as np

np.random.seed(420)


class Perzeptron:

    def __init__(self, layers, learning_rates=1):
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
            self.network[i]['bias'] = np.random.uniform(0, 1, (layers[i+1]))

        # self.bias = np.random.uniform(0, 1, (self.layer_count, 1))
        self.N, self.M = layers[0], layers[-1]


    # Convert a 1D array into a 2D array where the second dimension is 1
    #   Needed for matrix mulitplication and transposes
    def to_tensor(self, x):
        return np.reshape(x, (x.shape[0],1))

    def forward_propagate(self, inputs):
        #inputs = self.to_tensor(inputs)
        self.network[0]['inputs'] = self.to_tensor(inputs)
        # ACTIVATION FUNCTIONS DIFFERENT FOR HIDDEN LAYERS?
        for i in range(self.layer_count):
            inputs = np.dot(inputs, self.network[i]['weights']) + self.network[i]['bias']
            self.network[i]['preoutputs'] = inputs
            inputs = np.tanh(inputs)    # activation
            self.network[i]['outputs'] = inputs
        return inputs


    def back_propagate(self, outputs, expected):
        for i in reversed(range(0, self.layer_count)):
            if i != self.layer_count - 1:
                outputs = self.network[i]['outputs'] 
                delta_below = self.network[i+1]['delta']
                weights_below = self.network[i+1]['weights']
                print("Shapes!")
                print(delta_below.shape)
                print(weights_below.shape)
                print(self.network[i]['preoutputs'].shape)
                delta = np.multiply( weights_below.T * delta_below, self.transfer_derivative(self.network[i]['preoutputs']))
                print(delta.shape)
                self.network[i]['delta'] = delta
            else:   # runs first
                # output layer delta calc is at least correct
                #errors = outputs - expected
                errors = expected - outputs 
                delta = np.multiply(errors, self.transfer_derivative(self.network[i]['preoutputs']))
                self.network[i]['delta'] = delta    # store delta for backprop

            if i != 0:
                self.network[i]['bias'] += delta
                self.network[i]['weights'] += self.learning_rates * self.network[i-1]['outputs'] * delta.T
            else:
                self.network[i]['bias'] += delta
                self.network[i]['weights'] += self.learning_rates * self.network[0]['inputs']  * delta.T 


    def transfer_derivative(self, outputs):
        # tanh derivative
        derivative = 1 - outputs ** 2
        return derivative


    def train(self, input_data, teacher_data):
        for inputs, labels in list(zip(input_data, teacher_data)):
            for zzz in range(6):
                outputs = self.forward_propagate(inputs)
                error = outputs - labels
                self.print_error(error)
                self.back_propagate(outputs, labels)
                # self.print_everything(labels)
            return

    def print_error(self, error):
        print("Loss:  \t" + str(error))


    def print_everything(self, labels):
        print("---Inputs---")
        print("\t\tShape:   \n" + str(self.network[0]['inputs'].shape) )
        print("\t\tValues:   \n" + str(self.network[0]['inputs']) )
        print("\t\tLabels:   \n" + str(labels) )
        for i in range(self.layer_count):
            print("----- Layer " + str(i) + "------")
            print("\t\tShape:   \n" + str(self.network[i]['weights'].shape) )
            print("\t\tValues:   \n" + str(self.network[i]['weights']) )
            print("\t\tBias:   \n" + str(self.network[i]['bias']) )
            print("\t\tDelta:   \n" + str(self.network[i]['delta']) )
            print("\t\tBefore Activ:  \n" + str(self.network[i]['preoutputs']))
            print("\t\tOut:  \n" + str(self.network[i]['outputs']))
        print("")
        print("Error:  \n" + str(self.error))
