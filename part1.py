############
# Natnicha #
############
import numpy as np

class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output):
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_input = n_input
        self.training_size = 2
        self.w_layer1 = np.array([[0.1, 0.2], [0.1, 0.1]])
        self.w_layer2 = np.array([[0.1, 0.1], [0.1, 0.2]])
        self.b_layer1 = np.matrix("0.1 ; 0.1")
        self.b_layer1 = np.array(self.b_layer1)
        self.b_layer2 = np.matrix("0.1;0.1")
        self.b_layer2 = np.array(self.b_layer2)
        self.train_x = np.array([[0.1, 0.1], [0.1, 0.2]])
        self.train_y = np.array([[0, 1], [1, 0]])

    def train(self, n_epochs, minibatch_size, lr):
        indexes = []
        for i in range(self.training_size):
            indexes.append(i)
        indexes = np.array(indexes)
        for j in range(n_epochs):
            np.random.shuffle(indexes)
            minibatches = []
            minibatches = np.split(indexes, self.training_size)
            for batch in minibatches:
                for i in batch:
                    # forward prop
                    x = self.train_x[i]
                    x = x.reshape(self.n_input, 1)
                    y = self.train_y[i]
                    z2 = np.dot(self.w_layer1, x) + (1 * self.b_layer1)
                    z2 = np.array(z2)
                    a2 = sigmoid(z2)
                    z3 = np.dot(self.w_layer2, a2) + (1 * self.b_layer2)
                    a3 = sigmoid(z3)
                    # backprop
                    # layer 2
                    b_matrix2 = np.ones((self.n_output, 1))
                    delta3 = -(y - a3) * sigmoid_prime(z3)
                    dEdW2 = np.dot(a2.transpose(), delta3)
                    dEdB2 = np.dot(delta3, b_matrix2)

                    # layer 1
                    b_matrix1 = np.ones((self.n_hidden, 1))
                    delta2 = np.dot(delta3, self.w_layer2.transpose()) * sigmoid_prime(z2)
                    dEdW1 = np.dot(delta2, x)
                    dEdB1 = np.dot(delta2, b_matrix1)

                    # UPDATE WEIGHTS!!!
                    self.w_layer2 -= ((lr/minibatch_size) * dEdW2)
                    self.b_layer2 -= ((lr/minibatch_size) * dEdB2)

                    self.w_layer1 -= ((lr/minibatch_size) * dEdW1)
                    self.b_layer1 -= ((lr/minibatch_size) * dEdB1)

        print("Epoch: ", j+1)
        print("Updated Layer 2 weights:", self.w_layer2)
        print("Updated Layer 2 bias:", self.b_layer2)
        print("Updated Layer 1 weights:", self.w_layer1)
        print("Updated Layer 1 bias:", self.b_layer1)

    def test(self, bias_term=1):
        accurate = 0
        inaccurate = 0
        for i in range(self.test_size):
            x = self.test_x[i]
            y = self.test_y[i]
            z2 = np.dot(x, self.w_layer1) + (bias_term * self.b_layer1)
            a2 = sigmoid(z2)
            z3 = np.dot(a2, self.w_layer2) + (bias_term * self.b_layer2)
            a3 = sigmoid(z3)

            if self.test_y[i] == a3.argmax():
                accurate += 1
            else:
                inaccurate += 1
        return (accurate / self.test_size) * 100
    #creates new neural network each time
    def reset_synapse(self):
        np.random.seed(1)
        self.w_layer1 = 2 * np.random.rand(self.n_input, self.n_hidden) - 1
        self.w_layer2 = 2 * np.random.rand(self.n_hidden, self.n_output) - 1
        self.b_layer1 = 2 * np.random.rand(1, self.n_hidden) - 1
        self.b_layer2 = 2 * np.random.rand(1, self.n_output) - 1


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

#                 n_inputs, n_hidden, n_outputs
network = NeuralNetwork(2, 2, 2)
#       n_epochs, minibatch_size, lr
network.train(1, 2, 0.1)


