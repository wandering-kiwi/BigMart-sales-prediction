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
    def __init__(self, dimensions, activation=sigmoid, d_activation=d_sigmoid, learning_rate=0.00001):
        self.dimensions = dimensions
        self.connections = range(len(dimensions)-1)
        self.activation=activation
        self.d_activation=d_activation
        self.biases = [np.array([0.0 for j in range(i)]) for i in dimensions][1::]
        self.weights = self.init_weights()
        self.learning_rate = learning_rate
    def init_weights(self):
        return [2 * np.random.random((self.dimensions[i-1], self.dimensions[i])) - 1 for i in range(len(self.dimensions))][1::]
        # return [2 * np.zeros((self.dimensions[i-1], self.dimensions[i])) + 2.0 for i in range(len(self.dimensions))][1::]
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
                    # d_w = np.dot(d_z, np.repeat(A[i].reshape(1, A[i].size), d_z.size, axis=0))
                    # d_w = d_z * np.repeat(A[i].reshape(1, A[i].size), d_z.size, axis=0)
                    d_w = np.outer(A[i], d_z)
                    d_b = d_z
                    d_a = np.dot(self.weights[i], d_z)
                    actv_grad_vecs.insert(0, d_a)
                    weight_grad_mats[j].insert(0, d_w)
                    bias_grad_mats[j].insert(0, d_b)
            # for i in weight_grad_mats[j]:
            #     print(i)
        # print(weight_grad_mats)
        # print([np.mean(np.array([i[j] for i in weight_grad_mats]), axis=0) for j in self.connections], 'mean')
        # weight_grad_mats = np.array(weight_grad_mats)
        # bias_grad_mats = np.array(bias_grad_mats) 
        weight_grad_vecs =[np.mean(np.array([i[j] for i in weight_grad_mats]), axis=0) for j in self.connections]
        bias_grad_vecs = [np.mean(np.array([i[j] for i in bias_grad_mats]), axis=0) for j in self.connections]
        # for i in self.connections:
            
        #     # axis 0 means collapsing to a horizontal vector, axis 1 means to a vertical one.
        #     weight_grad_vecs.append(weight_grad_mats.sum(axis=0))
        #     print(weight_grad_mats.sum(axis=0), 'sum')
        #     bias_grad_vecs.append(bias_grad_mats.sum(axis=0)/ len(train_in))
        return weight_grad_vecs, bias_grad_vecs
    def train(self, train_in, train_out, iters):
        for j in range(iters):
            # make code about batches later
            weight_grad_vecs, bias_grad_vecs = self.back_prop(train_in, train_out)
            for i in self.connections:
                # print(self.weights[i]) 
                self.weights[i] -= weight_grad_vecs[i] * self.learning_rate
                self.biases[i] -= bias_grad_vecs[i] * self.learning_rate
        return self.cost(train_in, train_out)
# just for testing

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






X = Network(np.array([3, 4, 5, 2]), activation=temp, d_activation=d_temp)
train_in = np.array([
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3]
    ])
train_out = np.array([
    [1, 2],
    [1, 2],
    [1, 2],
    [1, 2]
    ])
# print(X.cost(train_in, train_out))
# 
# for i in X.connections:
#     print(Z[i], A[i])
# print(X.weights[0])
# Z, A = X.run(train_in[0], all_layers=True)

print(X.cost(train_in, train_out), 'cost')
print(X.train(train_in, train_out, 100))
print(X.run([1, 2, 3]))