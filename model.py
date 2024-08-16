import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def d_sigmoid(x):
    return x * (1 - x)
def ReLU(x):
    return x if x>0 else 0
def d_ReLU(x):
    return 1 if x>0 else 0

class Network():
    def __init__(self, dimensions, activation=sigmoid, d_activation=d_sigmoid, learning_rate=0.01):
        self.dimensions = dimensions
        self.connections = range(len(dimensions-1))
        self.activation=activation
        self.d_activation=d_activation
        self.biases = [np.array([0 for j in range(i)]) for i in dimensions][1::]
        self.weights = self.init_weights()
        self.learning_rate = learning_rate
    def init_weights(self):
        return [2 * np.random.random((self.dimensions[i-1], self.dimensions[i])) - 1 for i in range(len(self.dimensions))][1::]
    def cost(self, train_in, train_out):
        results = np.apply_along_axis(self.run, 1, train_in)
        results -= train_out
        results *= results/2
        result = np.sum(results, axis=1)
        avg = sum(result) / len(result)
        return avg
    def run(self, input, all_layers=False):
        Z = []
        A = []
        layer = np.array(input)
        layer = layer.reshape(1, layer.size)
        for i in self.connections:
            layer = np.dot(layer, self.weights[i]) + self.biases[i].reshape(1, self.biases[i].size)
            Z.append(layer[0])
            layer = self.activation(layer)
            A.append(layer[0])
        if all_layers:
            return Z, A
        return layer[0]
    def back_prop(self, train_in, train_out):
        weight_grad_mats = []
        bias_grad_mats = []
        for j in range(train_in):
            actv_grad_vecs=[]
            x = train_in[j]
            y=train_out[j]
            Z, A = self.run(x, all_layers=True)
            for j in reversed(range(len(self.dimensions))):
                if i== len(self.dimensions) -1:
                    d_z = A[i] - y
                    d_a= d_z * self.d_activation(Z[i])
                    actv_grad_vecs.append(d_a)
                else:
                    # haven't sorted out code yet in this secton
                    pass
        weight_grad_vecs = []
        bias_grad_vecs = []
        for i in self.connections:
            # axis 0 means collapsing to a horizontal vector, axis 1 means to a vertical one.
            weight_grad_vecs.append(weight_grad_mats.sum(axis=0))
            bias_grad_vecs.append(bias_grad_mats.sum(axis=0))
        return weight_grad_vecs, bias_grad_vecs
    def train(self, train_in, train_out, iters):
        for i in range(iters):
            # make code about batches later
            weight_grad_vecs, bias_grad_vecs = self.backprop(train_in, train_out)
            for i in self.connections:
                self.weights[i] += weight_grad_vecs[i] * self.learning_rate
                self.biases[i] += bias_grad_vecs[i] * self.learning_rate
        return self.cost(train_in, train_out)
# just for testing


X = Network(np.array([3, 4, 5, 2]))
train_in = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
    ])
train_out = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
    ])
# print(X.cost(train_in, train_out))
Z, A = X.run(train_in[0], all_layers=True)
for i in X.connections:
    print(Z[i], A[i])