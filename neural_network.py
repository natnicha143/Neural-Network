############
# Natnicha #
############
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output):
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_input = n_input
        self.train_x = np.loadtxt("TrainDigitX.csv.gz", delimiter=',')
        self.training_size = len(self.train_x)
        self.raw_y = np.loadtxt("TrainDigitY.csv.gz", dtype='int')
        # creates a np array of zeros, with length of training size and width of 10
        self.train_y = np.zeros((self.training_size, 10), dtype='int')

        self.test_x = np.loadtxt("TestDigitX.csv.gz", delimiter=',')
        self.test_size = len(self.test_x)
        self.test_y = np.loadtxt("TestDigitY.csv.gz", dtype='int')
        #convert to vector test label
        for i in range(self.training_size):
            self.train_y[i][self.raw_y[i]] = 1

        np.random.seed(1)
        self.w_layer1 = 2 * np.random.rand(n_input, n_hidden) - 1
        self.w_layer2 = 2 * np.random.rand(n_hidden, n_output) - 1
        self.b_layer1 = 2 * np.random.rand(1, n_hidden) - 1
        self.b_layer2 = 2 * np.random.rand(1, n_output) - 1

    def train(self, n_epochs, minibatch_size, lr):
        indexes = []
        for i in range(self.training_size):
            indexes.append(i)
        indexes = np.array(indexes)
        for j in range(n_epochs):
            print("Epoch:", j)
            np.random.shuffle(indexes)
            minibatches = []
            minibatches = np.split(indexes, self.training_size/20)
            for batch in minibatches:
                for i in batch:
                    # forward prop
                    x = self.train_x[i]
                    y = self.train_y[i]
                    z2 = np.dot(x, self.w_layer1) + (1 * self.b_layer1)
                    z2 = np.array(z2)
                    a2 = sigmoid(z2)
                    z3 = np.dot(a2, self.w_layer2) + (1 * self.b_layer2)
                    a3 = sigmoid(z3)
                    # backprop
                    # layer 2
                    #creates numpy array of ones for bias term
                    b_matrix2 = np.ones((self.n_output, 1), dtype='int')
                    delta3 = -(y-a3) * sigmoid_prime(z3)
                    dEdW2 = np.dot(a2.transpose(), delta3)
                    dEdB2 = np.dot(delta3, b_matrix2)

                    # layer 1
                    b_matrix1 = np.ones((self.n_hidden, 1), dtype='int')
                    delta2 = np.dot(delta3, self.w_layer2.transpose()) * sigmoid_prime(z2)
                    dEdW1 = np.dot(x.reshape(self.n_input, 1), delta2)
                    dEdB1 = np.dot(delta2, b_matrix1)

                    # UPDATE WEIGHTS!!!
                    #for layer 2
                    self.w_layer2 -= ((lr/minibatch_size) * dEdW2)
                    self.b_layer2 -= ((lr/minibatch_size) * dEdB2)
                    #for layer 1
                    self.w_layer1 -= ((lr/minibatch_size) * dEdW1)
                    self.b_layer1 -= ((lr/minibatch_size) * dEdB1)

    def test(self, bias_term=1):
        accurate = 0
        inaccurate = 0
        for i in range(self.test_size):
            x = self.test_x[i]
            z2 = np.dot(x, self.w_layer1) + (bias_term * self.b_layer1)
            a2 = sigmoid(z2)
            z3 = np.dot(a2, self.w_layer2) + (bias_term * self.b_layer2)
            a3 = sigmoid(z3)
            #if the output label is equal to the output by forward propagation
            if self.test_y[i] == a3.argmax():
                accurate += 1
            else:
                inaccurate += 1
        print((accurate / self.test_size) * 100, "%")
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
network = NeuralNetwork(784, 30, 10)
#       n_epochs, minibatch_size, lr
network.train(30, 20, 3)
network.test()


#For testing

# sample_mb_size = [1, 5, 10, 20, 100]
#
# learn_sample = [0.001, 0.1, 1.0, 10, 100]
#
# for i in learn_sample:
#     network.train(30, 20, i)
#     plt.plot(i, network.test(), "o", color='pink')
#     network.reset_synapse()
#
# plt.ylabel("Accuracy %")
# plt.xlabel("Learning Rate")
# plt.show()
