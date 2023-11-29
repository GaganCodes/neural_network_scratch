"""
Author: Gagandeep Randhawa

Code organization (separated by #++++++++++ and checkpoints):
1. Importing libraries (non-exhaustive)
2. Defining the dataset
3. Defining the neural network model and sklearn model
4. Running the optimization / training
5. Compare with scikit-learn
"""

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.neural_network import MLPRegressor

print("Checkpoint - Libraries imported.")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Defining the dataset, separating into training and validation set

np.random.seed(32)

BATCH_SIZE = 32
NUM_BATCH = 16
HIDDEN_DIM = 4
LEARNING_RATE = 0.01
MAX_EPOCHS = 1500
NUM_FEATURES = 3

X = np.random.randint(low=-20, high=20, size=(NUM_BATCH, BATCH_SIZE, NUM_FEATURES))
Y = np.random.rand(NUM_BATCH, BATCH_SIZE, 1)

print("Checkpoint - Dataset defined.")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Defining the neural network model
class NeuralNetwork:
    r"""
    For this example, we are looking at a one layered Neural Network with regression.
    Code is structured so the node numbers can be flexible for any future experiments.

    IMP: It is only done to understand training process (optimizing loss),
    not making any predictions to assess generalization error, could be a
    future extension.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 4,
        learn_rate: float = 0.001,
        max_epochs: int = 1000,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learn_rate = learn_rate
        self.max_epochs = max_epochs

        self.weight = dict()
        self.grad = dict()
        self.bias = dict()

        self.loss_history = []

        # Initializing the weight matrices to random small numbers
        self.weight["w1"] = 0.01 * np.random.randn(self.input_dim, self.hidden_dim)
        self.weight["w2"] = 0.01 * np.random.randn(self.hidden_dim, 1)
        self.weight["b1"] = np.zeros(self.hidden_dim)
        self.weight["b2"] = np.zeros(1)

        # Initializing the gradient matrices
        self.grad["w1"] = np.zeros((self.input_dim, self.hidden_dim))
        self.grad["w2"] = np.zeros((self.hidden_dim, 1))
        self.grad["b1"] = np.zeros(self.hidden_dim)
        self.grad["b2"] = np.zeros(1)

    def sigmoid(self, x) -> np.ndarray:
        # f(x) = 1/(1 + e^(-x))
        return 1.0 / (1.0 + np.exp(-x))

    def d_sigmoid(self, x) -> np.ndarray:
        # f'(x) = f(x)*(1-f(x))
        fx = self.sigmoid(x)
        return np.multiply(fx, 1 - fx)  # Element wise operation

    def calculate_loss(self, y_pred, y_true) -> float:
        # These are assumed to be of same length for simplicity in implementation
        return ((y_true - y_pred) ** 2).mean()

    def get_loss(self) -> list:
        # Doing this to discourage external manipulation
        return self.loss_history

    def update(self) -> None:
        for key in self.weight.keys():
            self.weight[key] = self.weight[key] - self.learn_rate * self.grad[key]

    def forward(self, x: np.ndarray, y: np.ndarray, training: bool = True) -> float:
        """
        Assuming there's batch training going on. N = batch_size
        """

        # Propagating the input forward
        h1 = np.matmul(x, self.weight["w1"]) + self.weight["b1"]  # Dim (N, hidden_dim)
        o1 = self.sigmoid(h1)  # Dim (N, hidden_dim)
        h2 = np.matmul(o1, self.weight["w2"]) + self.weight["b2"]  # Dim (N, 1)
        o2 = self.sigmoid(h2)  # Dim (N, 1)

        # Calculating loss / performance
        loss = self.calculate_loss(o2, y)  # Float

        # Back propagation
        if training:
            # Defining each progressive derivative for chain rule
            dl_do2 = 2 * (o2 - y)  # Dim (N, 1)
            do2_dh2 = self.d_sigmoid(h2)  # Dim (N, 1)
            dh2_dw2 = np.transpose(o1)  # Dim (hidden_dim, N)
            dh2_do1 = np.transpose(self.weight["w2"])  # Dim (1, hidden_dim)
            do1_dh1 = self.d_sigmoid(h1)  # Dim (N, hidden_dim)
            dh1_dw1 = x  # Dim (N, input_dim)

            # Applying chain rule for relevant gradient computation
            dl_dh2 = dl_do2 * do2_dh2  # Dim (N, 1)
            dl_w2 = np.matmul(dh2_dw2, dl_dh2)  # Dim (hidden_dim,)
            dl_dh1 = np.multiply(
                np.matmul(dl_dh2, dh2_do1), do1_dh1
            )  # Dim (N, hidden_dim)
            dl_w1 = np.matmul(
                np.transpose(dh1_dw1), dl_dh1
            )  # Dim (input_dim, hidden_dim)

            self.grad["w1"] = dl_w1
            self.grad["w2"] = dl_w2

            self.grad["b1"] = np.sum(dl_dh1, axis=0)
            self.grad["b2"] = np.sum(dl_dh2, axis=0)

        return loss

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        '''
        Assumption is that x_train and y_train have the right shape.
        '''
        for epoch in range(self.max_epochs):
            loss = 0.0
            for x, y in zip(x_train, y_train):
                loss += self.forward(x, y)
                self.update()

            # Recording the stats
            self.loss_history.append(loss / NUM_BATCH)


def plot_loss_curve(loss_self, t_self, loss_sklearn, t_sklearn) -> None:
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss Value")
    ax.set_title("Loss Curve", fontsize="large")

    ax.plot(loss_self, color="r", label=f"Self NN. t={1000*t_self:>0.1f}ms")
    ax.plot(loss_sklearn, color="b", label=f"Sklearn. t={1000*t_sklearn:>0.1f}ms")

    ax.set_ylim(ymax=1.1 * np.minimum(np.max(loss_self), np.max(loss_sklearn)))

    ax.legend()

    plt.show()


def plot_time_comparison(time_self, time_sklearn, batch) -> None:
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_xlabel("Total Batch Size")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Performance Comparison", fontsize="large")

    ax.plot(batch, time_self, 'ro-', label=f"Self NN")
    ax.plot(batch, time_sklearn, 'bo-', label=f"Sklearn")

    ax.legend()

    plt.show()


print("Checkpoint - Model defined.")
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Comparing performance

def compare_basic_performance():
    sklearn_model = MLPRegressor(
        hidden_layer_sizes=(HIDDEN_DIM,),
        activation="logistic",
        solver="sgd",
        batch_size=BATCH_SIZE,
        learning_rate_init=LEARNING_RATE,
        max_iter=MAX_EPOCHS,
        n_iter_no_change=MAX_EPOCHS
    )

    self_model = NeuralNetwork(
        input_dim=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
        learn_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
    )

    t1 = time.time()
    self_model.train(X, Y)
    t2 = time.time()
    sklearn_model.fit(X.reshape(NUM_BATCH * BATCH_SIZE, NUM_FEATURES), Y.reshape(NUM_BATCH * BATCH_SIZE))
    t3 = time.time()
    
    plot_loss_curve(self_model.get_loss(), t2 - t1, sklearn_model.loss_curve_, t3 - t2)

def compare_data_size_performance():
    sklearn_model = MLPRegressor(
        hidden_layer_sizes=(HIDDEN_DIM,),
        activation="logistic",
        solver="sgd",
        batch_size=BATCH_SIZE,
        learning_rate_init=LEARNING_RATE,
        max_iter=MAX_EPOCHS,
        n_iter_no_change=MAX_EPOCHS
    )

    self_model = NeuralNetwork(
        input_dim=NUM_FEATURES,
        hidden_dim=HIDDEN_DIM,
        learn_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
    )

    batch_size_exp = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    time_self = []
    time_sklearn = []

    for batch_num in batch_size_exp:
        print(f"Num of Batches: {batch_num}")
        x_try = np.random.randint(low=-20, high=20, size=(batch_num, BATCH_SIZE, NUM_FEATURES))
        y_try = np.random.rand(batch_num, BATCH_SIZE, 1)

        t1 = time.time()
        self_model.train(x_try, y_try)
        t2 = time.time()
        
        x_skl = x_try.reshape(batch_num * BATCH_SIZE, NUM_FEATURES)
        y_skl = y_try.reshape(batch_num * BATCH_SIZE)

        t3 = time.time()
        sklearn_model.fit(x_skl, y_skl)
        t4 = time.time()

        time_self.append(t2-t1)
        time_sklearn.append(t4-t3)

    plot_time_comparison(time_self, time_sklearn, batch_size_exp)

if __name__ == "__main__":
    compare_basic_performance()
    compare_data_size_performance()
