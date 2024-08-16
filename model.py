import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def d_sigmoid(x):
    return x * (1 - x)
def ReLU(x):
    return np.maximum(0, x)
def d_ReLU(x):
    return (x > 0).astype(float)
def temp(x):
    return 3*x
def d_temp(x):
    return x - x + 3

class Network():
    def __init__(self, dimensions, activation=sigmoid, d_activation=d_sigmoid, learning_rate=0.0001):
        self.dimensions = dimensions
        self.connections = range(len(dimensions)-1)
        self.activation=activation
        self.d_activation=d_activation
        self.biases = [np.array([0.0 for j in range(i)]) for i in dimensions][1::]
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
        A.append(layer[0])
        for i in self.connections:
            # print(i)
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
        for j in range(len(train_in)):
            weight_grad_mats.append([])
            bias_grad_mats.append([])
            actv_grad_vecs=[]
            x = train_in[j]
            y=train_out[j]
            Z, A = self.run(x, all_layers=True)
            for i in reversed(range(len(self.dimensions))):
                if i== len(self.dimensions) -1:
                    d_a = A[i] - y
                    actv_grad_vecs.append(d_a)
                else:
                    d_z = self.d_activation(Z[i]) * actv_grad_vecs[0]
                    d_w = np.outer(A[i], d_z)
                    d_b = d_z
                    d_a = np.dot(self.weights[i], d_z)
                    actv_grad_vecs.insert(0, d_a)
                    weight_grad_mats[j].insert(0, d_w)
                    bias_grad_mats[j].insert(0, d_b)
        weight_grad_vecs =[np.mean(np.array([i[j] for i in weight_grad_mats]), axis=0) for j in self.connections]
        bias_grad_vecs = [np.mean(np.array([i[j] for i in bias_grad_mats]), axis=0) for j in self.connections]
        return weight_grad_vecs, bias_grad_vecs
    def train(self, train_in, train_out, iters):
        for j in range(iters):
            # make code about batches later
            weight_grad_vecs, bias_grad_vecs = self.back_prop(train_in, train_out)
            for i in self.connections:
                self.weights[i] -= weight_grad_vecs[i] * self.learning_rate
                self.biases[i] -= bias_grad_vecs[i] * self.learning_rate
        return self.cost(train_in, train_out)
# just for testing


# simple network

# Y = Network(np.array([1, 1, 1, 1]), activation=temp, d_activation=d_temp)
# # Y = Network(np.array([1, 1, 1, 1]), activation=ReLU, d_activation=d_ReLU)
# train_in = np.array([
#     [1],
#     [1],
#     [1],
#     [1]
# ])
# train_out = np.array([
#     [2],
#     [2],
#     [2],
#     [2]
# ])
# print(Y.cost(train_in, train_out))
# print(Y.train(train_in, train_out, 100))
# print(Y.run([1]))




# more complex netork

learning_rate = 0.000003
iters = 2000
data_count = 70

X = Network(np.array([2, 7, 8, 1]), activation=temp, d_activation=d_temp, learning_rate = learning_rate)
train_in = []
train_out = []

for i in range(data_count):
    a = np.random.random()*10
    b = np.random.random()*10
    c = a + b
    train_in.append([a, b])
    train_out.append([c])
train_in = np.array(train_in)
train_out = np.array(train_out)
print(X.cost(train_in, train_out), 'cost')
new_cost = X.train(train_in, train_out, iters)
print(new_cost, 'improved cost')
print('2 + 2 =', X.run([2, 2])[0])

# after 100 lines of code, finally I can add 2 + 2