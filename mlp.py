import numpy as np

np.random.seed(420)


class Perzeptron:

    def __init__(self, layers, learning_rates=0.1):
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
            self.network[i]['bias'] = np.random.uniform(0, 1, (layers[i+1],1))

        # self.bias = np.random.uniform(0, 1, (self.layer_count, 1))
        self.N, self.M = layers[0], layers[-1]


    # Convert a 1D array into a 2D array where the second dimension is 1
    #   Needed for matrix mulitplication and transposes
    def to_tensor(self, x):
        return np.reshape(x, (x.shape[0],1))

    def forward_propagate(self, inputs):
        inputs = self.to_tensor(inputs)
        self.network[0]['inputs'] = self.to_tensor(inputs)
        # ACTIVATION FUNCTIONS DIFFERENT FOR HIDDEN LAYERS?
        for i in range(self.layer_count):
            inputs = np.dot(self.network[i]['weights'].T, inputs) + self.network[i]['bias']
            self.network[i]['preoutputs'] = inputs
            inputs = np.tanh(inputs)    # activation
            self.network[i]['outputs'] = self.to_tensor(inputs)
        # print("Forward Ho")
        # print(inputs.shape)
        return inputs


    def back_propagate(self, outputs, expected):
        for i in reversed(range(0, self.layer_count)):
            # print("------Run:  " + str(i) + "-----------")
            if i != self.layer_count - 1:
                outputs = self.network[i]['outputs'] 
                delta_below = self.to_tensor(self.network[i+1]['delta'])
                weights_below = self.network[i+1]['weights']
                # print("Shapes!")
                # print(delta_below.shape)
                # print(weights_below.shape)
                # print("Delta Below")
                # print(delta_below.shape)
                # print(delta_below)
                # print("Weights Below")
                # print(weights_below.shape)
                # print(weights_below)
                
                temp = np.dot(weights_below, delta_below)
                # print("Temp")
                # print(temp.shape)
                # print(temp)
                temp2 = self.transfer_derivative(self.network[i]['preoutputs'])
                # print("Preoutputs:")
                # print(temp2.shape)
                # print(temp2)

                # delta = np.zeros(temp.shape)
                # for i in range(temp.shape[0]):
                #     delta[i] = temp[i] *

                delta = np.multiply( temp, self.to_tensor(temp2))
                # print("Delta")
                # print(delta.shape)
                # print(delta)
                self.network[i]['delta'] = delta
            else:   # runs first
                # output layer delta calc is at least correct
                #errors = outputs - expected
                errors = expected - outputs 
                # print("Preoutputs:")
                # print(self.transfer_derivative(self.network[i]['preoutputs']).shape)
                # print(self.transfer_derivative(self.network[i]['preoutputs']))
                # print("Errors:")
                # print(errors.shape)
                # print(errors)
                delta = np.multiply(errors, self.transfer_derivative(self.network[i]['preoutputs']))
                # print("Delta:")
                # print(delta.shape)
                # print(delta)
                self.network[i]['delta'] = delta    # store delta for backprop

            if i != 0:
                # print("Deies Shapes")
                # print(self.network[i-1]['outputs'].shape)
                # print(delta.shape)
                # print(self.network[i]['weights'].shape)
                # print(delta)
                self.network[i]['bias'] += delta
                self.network[i]['weights'] +=  self.learning_rates * self.network[i-1]['outputs'] * self.to_tensor(delta).T
            else:
                # print("backup shapes")
                # print(delta.shape)
                # print(self.network[i]['bias'].shape)
                # print(self.network[0]['inputs'].shape)
                self.network[i]['bias'] += delta
                self.network[i]['weights'] += self.learning_rates * self.network[0]['inputs']  * delta.T 


    def transfer_derivative(self, outputs):
        # tanh derivative
        derivative = 1.0 - np.tanh(outputs) ** 2
        return derivative


    def train(self, input_data, teacher_data):
        for inputs, labels in list(zip(input_data, teacher_data)):
            for zzz in range(15):
                inputs = self.to_tensor(inputs)
                labels = self.to_tensor(labels)
                outputs = self.forward_propagate(inputs)

                error = outputs - labels
                self.print_error(error, labels, outputs)
                self.back_propagate(outputs, labels)
                # self.print_everything(labels)
            return

    def print_error(self, error, labels, outputs):
        print("---------------------------------------")
        print("Labels:  \n" + str(labels))
        print("Outputs:  \n" + str(outputs))
        print("Loss:  \n" + str(error))
        print("---------------------------------------")


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
