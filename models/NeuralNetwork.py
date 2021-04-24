import numpy as np
import pandas as pd

class NeuralNetwork():
    def __init__(self,x_no,y_no,h_no):
        '''
        i_size: size of input layer
        h_size: size of hidden layer
        o_size: size of output layer
        '''

        self.i_size = x_no
        self.o_size = y_no
        self.h_size = h_no


        self.weights = ["b1","W1","b2","W2"]

        self.params = {
                    "b1": np.zeros(shape=(self.h_size,1)),
                    "W1": np.random.randn(self.h_size, self.i_size) / (np.sqrt(self.i_size)),
                    "b2": np.zeros(shape=(self.o_size,1)),
                    "W2": np.random.randn(self.o_size, self.h_size) / (np.sqrt(self.h_size)),
                    }

        self.grads = {
                    "b1": np.zeros(shape=(self.h_size,1)),
                    "W1": np.zeros(shape=(self.h_size, self.i_size)),
                    "b2": np.zeros(shape=(self.o_size,1)),
                    "W2": np.zeros(shape=(self.o_size, self.h_size)),
                    }

        self.z = {
                    "b1": np.zeros(shape=(self.h_size,1)),
                    "W1": np.zeros(shape=(self.h_size, self.i_size)),
                    "b2": np.zeros(shape=(self.o_size,1)),
                    "W2": np.zeros(shape=(self.o_size, self.h_size)),
                    }

        self.o = {
                    "b1": np.zeros(shape=(self.h_size,1)),
                    "W1": np.zeros(shape=(self.h_size, self.i_size)),
                    "b2": np.zeros(shape=(self.o_size,1)),
                    "W2": np.zeros(shape=(self.o_size, self.h_size)),
                    }

        self.grads_prev = {
                    "b1": np.ones(shape=(self.h_size,1)),
                    "W1": np.ones(shape=(self.h_size, self.i_size)),
                    "b2": np.ones(shape=(self.o_size,1)),
                    "W2": np.ones(shape=(self.o_size, self.h_size)),
                    }

        self.step = {
                    "b1": np.ones(shape=(self.h_size,1)),
                    "W1": np.ones(shape=(self.h_size, self.i_size)),
                    "b2": np.ones(shape=(self.o_size,1)),
                    "W2": np.ones(shape=(self.o_size, self.h_size)),
                    }


        """# Bias 1 (rows = hidden size)
        self.b1 = np.zeros(shape=(self.h_size,1))

        # Weight matrix 1 (rows = hidden size; columns = input size)
        self.W1 = np.random.randn(self.h_size, self.i_size) / (np.sqrt(self.i_size))


        # Bias 2 (rows = output size)
        self.b2 = np.zeros(shape=(self.o_size,1))

        # Weight matrix 2 (rows = output size; columns = hidden size)
        self.W2 = np.random.randn(self.o_size, self.h_size) / (np.sqrt(self.h_size))"""



    def forward(self, X):

        # First layer
        self.A1 = X.T

        print(self.params["W1"],self.A1)

        # Dot product of X (input) and W1 + Bias
        self.Z2 = np.dot(self.params["W1"],self.A1) + self.params["b1"]

        # First activation function (Second layer)
        self.A2 = self.sigmoid(self.Z2)

        # Dot product of Hidden layer and W2 + Bias
        self.Z3 = np.dot(self.params["W2"],self.A2) + self.params["b2"]

        # Final activation function (Final layer)
        self.A3 = self.sigmoid(self.Z3)

        return self.A3

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)

    def backward(self, Y):

        m = Y.shape[0]

        self.dCost = (1 / m) * (self.A3 - Y.T)
        self.Delta3 = self.dCost * self.sigmoidPrime(self.A3)

        self.grads["W2"] = np.dot(self.Delta3, self.A2.T)
        self.grads["b2"] = np.sum(self.Delta3, axis=1, keepdims=True)
        dA2 = np.dot(self.params["W2"].T,self.Delta3)

        self.Delta2 = dA2 * self.sigmoidPrime(self.A2)

        self.grads["W1"] = np.dot(self.Delta2,self.A1.T)
        self.grads["b1"] = np.sum(self.Delta2, axis=1, keepdims=True)

        #print("\ndW1\n",self.dW1,"\ndb1\n", self.db1,"\ndW2\n", self.dW2,"\ndb2\n", self.db2)

        return self.grads["W2"],self.grads["b2"],self.grads["W1"],self.grads["b1"]

    def cost_comp(self,Y):

        m = Y.shape[0]

        return (1 / 2*m) * np.sum(np.square(Y.T - self.A3))

    def predict(self, X, Y):

        n = X.shape[0]
        p = np.zeros((1, n))

        prob = self.forward(X)

        for i in range(0, prob.shape[1]):
            if prob[0, i] > 0.5:  # 0.5 is threshold
                p[0, i] = 1
            else:
                p[0, i] = 0

        accuracy = np.sum(p[0] == Y) / n * 100

        return prob, p[0], accuracy

    def gd(self, lr=1):

        for w in self.weights:
            self.params[w] -= lr * self.grads[w]


    def gd_m(self, lr=0.01, decay=0.9):

        for w in self.weights:
            self.z[w] = decay * self.z[w] - lr * self.grads[w]
            self.params[w] += self.z[w]

    def rprop(self, inc=1.2, dec=0.5, step_sizes=(0.0001,50)):

        for w in self.weights:

            above_zero = (self.grads[w] > 0) * (self.grads_prev[w] > 0)

            for i in range(len(self.params[w])):
                for j in range(len(self.params[w][0])):

                    if above_zero[i,j]:
                        self.step[w][i,j] = min(self.step[w][i,j]*inc, step_sizes[1])
                    else:
                        self.step[w][i,j] = max(self.step[w][i,j]*dec, step_sizes[0])

                    self.params[w][i,j] -= np.sign(self.grads[w][i,j]) * self.step[w][i,j]

        self.grads_prev = self.grads

    def rmsprop(self, lr=0.01, decay=0.9, eps=0.0001):

        for w in self.weights:
            for i in range(len(self.params[w])):
                for j in range(len(self.params[w][0])):

                    self.z[w][i,j] = decay * self.z[w][i,j] + (1 - decay) * self.grads[w][i,j]
                    self.params[w][i,j] -= lr / (np.sqrt(abs(self.z[w][i,j]))+eps) * self.grads[w][i,j]

    def adam(self, t, lr=0.01, b=(0.9,0.999), eps=0.0001):

        for w in self.weights:
            for i in range(len(self.params[w])):
                for j in range(len(self.params[w][0])):

                    self.z[w][i,j] = b[0] * self.z[w][i,j] + (1 - b[0]) * self.grads[w][i,j]
                    self.o[w][i,j] = b[1] * self.o[w][i,j] + (1 - b[1]) * (self.grads[w][i,j]**2)

                    z_corr = self.z[w][i,j] / (1 - (b[0] ** t))
                    o_corr = self.o[w][i,j] / (1 - (b[1] ** t))

                    self.params[w][i,j] -= lr * z_corr / (np.sqrt(abs(o_corr)) + eps)

    def wame(self, lr=0.01, inc=1.2, dec=0.1, a=0.9, step_sizes=(0.01,100), eps=0.0001):

        for w in self.weights:

            above_zero = (self.grads[w] > 0) * (self.grads_prev[w] > 0)

            for i in range(len(self.params[w])):
                for j in range(len(self.params[w][0])):

                    if above_zero[i,j]:
                        self.step[w][i,j] = min(self.step[w][i,j]*inc, step_sizes[1])
                    else:
                        self.step[w][i,j] = max(self.step[w][i,j]*dec, step_sizes[0])

                    self.z[w][i,j] = a * self.z[w][i,j] + (1 - a) * self.step[w][i,j]
                    self.o[w][i,j] = a * self.o[w][i,j] + (1 - a) * (self.grads[w][i,j]**2)

                    #print(self.params[w][i,j],lr,self.grads[w][i,j],self.o[w][i,j],self.z[w][i,j])

                    self.params[w][i,j] -= lr * self.grads[w][i,j] / (self.o[w][i,j] * self.z[w][i,j] + eps)

        self.grads_prev = self.grads


    def train(self, X, Y, epochs, optimizer):

        cost_log = []
        for t in range(epochs):
            self.forward(X)
            self.backward(Y)
            cost = self.cost_comp(Y)
            self.GD()
            cost_log.append([t,cost])

            if t % 500 == 0:
                print(f"Loss after iteration {t}: {round(cost,5)}")

        cost_log = pd.DataFrame(cost_log, columns=["epochs","loss"])

        return cost_log
