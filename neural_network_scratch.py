# Importing relevant libraries
import numpy as np

# Defining sigmoid function
def sigmoid(x):
    # f(x) = 1/(1 + e^(-x))
    return 1/(1+np.exp(-x))

# Defining derivative of sigmoid
def d_sigmoid(x):
    # f'(x) = f(x)*(1-f(x))
    return sigmoid(x)*(1 - sigmoid(x))

# Defining loss function
def mse_loss(y_true, y_pred):
    # These are assumed to be of same length for simplicity in implementation
    return ((y_true - y_pred)**2).mean()

# Defining Neuron
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

# Defining NeuralNetwork structure
class NeuralNetwork:
    def __init__(self, learn_rate=0.1, epochs=5000):
        self.learn_rate = learn_rate
        self.epochs = epochs

        # Defining weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Defining biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

        # Storing the loss curve information
        self.loss = []
    
    def feedforward(self, x):
        # x is an input numpy array with 2 elements
        # h1 and h2 are hidden nodes
        h1 = sigmoid(self.w1*x[0] + self.w2*x[1] + self.b1)
        h2 = sigmoid(self.w3*x[0] + self.w4*x[1] + self.b1)

        o1 = sigmoid(self.w5*h1 + self.w6*h2 + self.b3)

        return o1

    def train(self, data, all_y_trues):
        # data = assumed to be (n_samples, 2) numpy array
        # all_y_trues = assumed to be (n_samples, ) numpy array

        for epoch in range(self.epochs):
            for x, y_true in zip(data, all_y_trues):
                # Doing feedforward
                sum_h1 = self.w1*x[0] + self.w2*x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3*x[0] + self.w4*x[1] + self.b1
                h2 = sigmoid(sum_h2)
                
                sum_o1 = self.w5*x[0] + self.w6*x[1] + self.b1
                o1 = sigmoid(sum_o1)
                
                y_pred = o1

                # Calculating required derivatives
                # Naming: dL_dw = partial derivative of L wrt w
                dL_dypred = -2*(y_true-y_pred)

                # Node o1
                dypred_dw5 = h1*d_sigmoid(sum_o1)
                dypred_dw6 = h2*d_sigmoid(sum_o1)
                dypred_db3 = d_sigmoid(sum_o1)
                
                dypred_h1 = self.w5*d_sigmoid(sum_o1)
                dypred_h2 = self.w6*d_sigmoid(sum_o1)
                
                # Node h1
                dh1_dw1 = x[0]*d_sigmoid(sum_h1)
                dh1_dw2 = x[1]*d_sigmoid(sum_h1)
                dh1_db1 = d_sigmoid(sum_h1)
                
                # Node h2
                dh2_dw3 = x[0]*d_sigmoid(sum_h2)
                dh2_dw4 = x[1]*d_sigmoid(sum_h2)
                dh2_db2 = d_sigmoid(sum_h2)

                # Updating weights and biases
                # Node h1
                self.w1 -= self.learn_rate*dL_dypred*dypred_h1*dh1_dw1
                self.w2 -= self.learn_rate*dL_dypred*dypred_h1*dh1_dw2
                self.b1 -= self.learn_rate*dL_dypred*dypred_h1*dh1_db1

                # Node h2
                self.w3 -= self.learn_rate*dL_dypred*dypred_h2*dh2_dw3
                self.w4 -= self.learn_rate*dL_dypred*dypred_h2*dh2_dw4
                self.b2 -= self.learn_rate*dL_dypred*dypred_h2*dh2_db2

                # Node o1
                self.w5 -= self.learn_rate*dL_dypred*dypred_dw5
                self.w6 -= self.learn_rate*dL_dypred*dypred_dw6
                self.b3 -= self.learn_rate*dL_dypred*dypred_db3

            # Calculating total loss at the end of each epoch
            y_preds = np.apply_along_axis(self.feedforward, 1, data)
            self.loss.append(mse_loss(all_y_trues, y_preds))
        
        def get_loss(self):
            return np.array(self.loss)

# Running the problem
data = np.array([
    [  2, -16],
    [ 21, -17],
    [-20, -20],
    [-17, -19],
    [ 16,  14],
    [ 24,  -2],
    [  0,  11],
    [ 17, -22],
    [ -4,   5],
    [-15,  -3],
    [-21,  12],
    [ -3,  -7],
    [-12,  16],
    [  5,   2],
    [-10, -25],
    [-18, -17],
    [ -2, -15],
    [ -6, -15],
    [ -7, -12],
    [ -3,   8],
    [ 11, -24],
    [ 13,  11],
    [  6, -12],
    [ 14,   3],
    [ 15, -19],
    [-18, -10],
    [ -2,  20],
    [  0, -17],
    [  6,  16],
    [-17,   1],
    [ 10,  -6],
    [-16, -10],
    [ -6,  15],
    [  9,   1],
    [  8,  22],
    [-10,  17],
    [ -5,   3],
    [ -1,  -5],
    [ -6,  24],
    [  1,   3],
    [  3,   0],
    [ 22,   0],
    [ 17, -11],
    [-21,  13],
    [ 18, -19],
    [ 16,  -2],
    [-17,   7],
    [ 24,  13],
    [ 10,  -5],
    [ 19,  -6]])

all_y_trues = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0,
       1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1,
       1, 0, 1, 1, 1, 0])


# Comparing with Scikit-Learn
from sklearn.neural_network import MLPClassifier
import time

clf = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic',
                    learning_rate_init=0.1, max_iter=5000, random_state=7,
                    tol=1e-10, early_stopping=False)

# Scikit Learn
t1 = time.time()
clf.fit(data, all_y_trues)
t2 = time.time()
time_sklean = t2-t1

# Hard-coded neural network
network = NeuralNetwork()
t1 = time.time()
network.train(data, all_y_trues)
t2 = time.time()
time_NN = t2-t1

# Plotting comparison results
import matplotlib.pyplot as plt
figures, ax = plt.subplots()
ax.grid()
ax.plot(clf.loss_curve_, color='r', label="SKL, t=%.1fms"%(1000*time_sklean))
ax.plot(network.loss, color='g', label="NN, t=%.1fms"%(1000*time_NN))
ax.set_xlabel("Number of Iterations", fontsize="large")
ax.set_ylabel("Loss", fontsize="large")
ax.legend()
ax.set_title("Loss Curve", fontsize="x-large")

plt.show()