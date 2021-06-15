import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn

class NeuralNetwork():
    def __init__(self,input_size, hidden_layer_size = 3, output_size = 1):
        '''
        i_size: size of input layer
        h_size: size of hidden layer
        o_size: size of output layer
        '''

        self.i_size = input_size
        self.h_size = hidden_layer_size
        self.o_size = output_size

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

        self.grads_prev = {
                    "b1": np.ones(shape=(self.h_size,1)),
                    "W1": np.ones(shape=(self.h_size, self.i_size)),
                    "b2": np.ones(shape=(self.o_size,1)),
                    "W2": np.ones(shape=(self.o_size, self.h_size)),
                    }

        self.z = copy.deepcopy(self.grads)
        self.o = copy.deepcopy(self.grads)
        self.step = copy.deepcopy(self.grads_prev)

        self.t = 1

    def forward(self, X):

        # First layer
        self.A1 = X.T

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

    def backward(self, Y, f):


        self.dCost = f(Y.T)

        self.Delta3 = self.dCost * self.sigmoidPrime(self.A3)

        self.grads["W2"] = np.dot(self.Delta3, self.A2.T)
        self.grads["b2"] = np.sum(self.Delta3, axis=1, keepdims=True)
        dA2 = np.dot(self.params["W2"].T,self.Delta3)

        self.Delta2 = dA2 * self.sigmoidPrime(self.A2)

        self.grads["W1"] = np.dot(self.Delta2,self.A1.T)
        self.grads["b1"] = np.sum(self.Delta2, axis=1, keepdims=True)

        #print("\ndW1\n",self.dW1,"\ndb1\n", self.db1,"\ndW2\n", self.dW2,"\ndb2\n", self.db2)

        return self.grads["W2"],self.grads["b2"],self.grads["W1"],self.grads["b1"]

    def mse(self,Y):
        m = Y.shape[0]
        return (1 / 2*m) * np.sum(np.square(Y.T - self.A3))

    def d_mse(self,Y):
        """
        Y: Target  (format [[0,0,0,0]] )
        """

        m = Y.shape[1]
        return (1 / m) * (self.A3 - Y)

    def b_cross_entropy(self, y, P):
        """
        Binary Cross Entropy
        P: Estimated probability of belonging to class 1
        Y: Target
        """

        m = y.shape[0]
        ce = -y * np.log(P) - (1 - y) * np.log(1 - P)
        loss = ce.sum() / ce.shape[1]

        return ce, loss


    def db_cross_entropy(self,y):
        """
        Derivative of Binary Cross Entropy
        P: Estimated probability of belonging to class 1
        Y: Target
        """
        P = self.A3
        return (-y / P) + (1 - y)/(1 - P)


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

    def gd(self, attr):

        lr = attr["lr"]

        for w in self.weights:
            self.params[w] -= lr * self.grads[w]


    def gd_m(self, attr):

        for w in self.weights:
            self.z[w] = attr["decay"] * self.z[w] - attr["lr"] * self.grads[w]
            self.params[w] += self.z[w]

    def rprop(self, attr):

        for w in self.weights:

            above_zero = (self.grads[w] > 0) * (self.grads_prev[w] > 0)

            for i in range(len(self.params[w])):
                for j in range(len(self.params[w][0])):

                    if above_zero[i,j]:
                        self.step[w][i,j] = min(self.step[w][i,j]*attr["inc"], attr["step_sizes"][1])
                    else:
                        self.step[w][i,j] = max(self.step[w][i,j]*attr["dec"], attr["step_sizes"][0])

                    self.params[w][i,j] -= np.sign(self.grads[w][i,j]) * self.step[w][i,j]

        self.grads_prev = self.grads

    def rmsprop(self, attr):

        for w in self.weights:
            for i in range(len(self.params[w])):
                for j in range(len(self.params[w][0])):

                    self.z[w][i,j] = attr["decay"] * self.z[w][i,j] + (1 - attr["decay"]) * self.grads[w][i,j]
                    self.params[w][i,j] -= attr["lr"] / (np.sqrt(abs(self.z[w][i,j]))+attr["eps"]) * self.grads[w][i,j]

    def adam(self, attr):

        for w in self.weights:
            for i in range(len(self.params[w])):
                for j in range(len(self.params[w][0])):

                    self.z[w][i,j] = attr["b"][0] * self.z[w][i,j] + (1 - attr["b"][0]) * self.grads[w][i,j]
                    self.o[w][i,j] = attr["b"][1] * self.o[w][i,j] + (1 - attr["b"][1]) * (self.grads[w][i,j]**2)

                    z_corr = self.z[w][i,j] / (1 - (attr["b"][0] ** self.t))
                    o_corr = self.o[w][i,j] / (1 - (attr["b"][1] ** self.t))

                    self.params[w][i,j] -= attr["lr"] * z_corr / (np.sqrt(abs(o_corr)) + attr["eps"])
            self.t += 1

    def wame1(self, attr):

        for w in self.weights:


            above_zero = (self.grads[w] > 0) * (self.grads_prev[w] > 0)

            for i in range(len(self.params[w])):
                for j in range(len(self.params[w][0])):

                    if above_zero[i,j]:
                        self.step[w][i,j] = min(self.step[w][i,j]*attr["inc"], attr["step_sizes"][1])
                    else:
                        self.step[w][i,j] = max(self.step[w][i,j]*attr["dec"], attr["step_sizes"][0])

                    self.z[w][i,j] = attr["a"] * self.z[w][i,j] + (1 - attr["a"]) * self.step[w][i,j]
                    self.o[w][i,j] = attr["a"] * self.o[w][i,j] + (1 - attr["a"]) * (self.grads[w][i,j]**2)

                    delta = attr["lr"] * self.grads[w][i,j] / (self.o[w][i,j]) * self.z[w][i,j]

                    self.params[w][i,j] -= delta


        self.grads_prev = self.grads

    def wame2(self, attr):

        for w in self.weights:


            above_zero = (self.grads[w] > 0) * (self.grads_prev[w] > 0)

            for i in range(len(self.params[w])):
                for j in range(len(self.params[w][0])):

                    if above_zero[i,j]:
                        self.step[w][i,j] = min(self.step[w][i,j]*attr["inc"], attr["step_sizes"][1])
                    else:
                        self.step[w][i,j] = max(self.step[w][i,j]*attr["dec"], attr["step_sizes"][0])

                    self.z[w][i,j] = attr["a"] * self.z[w][i,j] + (1 - attr["a"]) * self.step[w][i,j]
                    self.o[w][i,j] = attr["a"] * self.o[w][i,j] + (1 - attr["a"]) * (self.grads[w][i,j]**2)

                    delta = attr["lr"] * self.grads[w][i,j] / (self.o[w][i,j] * self.z[w][i,j])

                    self.params[w][i,j] -= delta


        self.grads_prev = self.grads




class LinearRegression():
    def __init__(self):
        # Initialise the parameters
        self.theta = [np.random.uniform(0,1,1) for _ in range(2)]
        self.derivatives = [self.dx_theta0, self.dx_theta1]

        self.z = [0,0]
        self.o = [0,0]
        self.grads_prev = [0,0]
        self.step = [1,1]

        self.t = 1

    def yhat(self,x):
        return self.theta[0] + self.theta[1]*x

    def dx_theta0(self,x,y):
        m = len(x)
        dx = -2*(y - self.yhat(x))
        gr = dx.sum() / m
        return dx, gr

    def dx_theta1(self,x,y):
        m = len(x)
        dx = -2*x*(y - self.yhat(x))
        gr = dx.sum() / m
        return dx, gr

    def mse(self,X,Y):
        residuals = Y - self.yhat(X)
        RSS = (residuals**2).sum()
        return RSS/len(X)

    def gd(self,xs,ys,attr):
        gr = [0,0]

        for i, dx in enumerate(self.derivatives):

            _, gr[i] = dx(xs,ys)
            self.theta[i] -= attr["lr"]*gr[i]


    def gd_m(self,xs,ys,attr):

        gr = [0,0]
        for i, dx in enumerate(self.derivatives):

            _, gr[i] = dx(xs,ys)

            self.z[i] = attr["decay"] * self.z[i] - attr["lr"] * gr[i]
            self.theta[i] += self.z[i]

    def rprop(self,xs,ys,attr):

        gr = [0,0]

        for i, dx in enumerate(self.derivatives):

            _, gr[i] = dx(xs,ys)

            if gr[i] * self.grads_prev[i] > 0:
                self.step[i] = min(self.step[i] * attr["inc"], attr["step_sizes"][1])

            elif gr[i] * self.grads_prev[i] < 0:
                self.step[i] = max(self.step[i] * attr["dec"], attr["step_sizes"][0])

            self.theta[i] -= np.sign(gr[i]) * self.step[i]

        self.grads_prev = gr

    def rmsprop(self,xs,ys,attr):

        gr = [0,0]
        for i, dx in enumerate(self.derivatives):

            _, gr[i] = dx(xs,ys)

            self.z[i] = attr["decay"] * self.z[i] + (1-attr["decay"])* (gr[i] **2)

            self.theta[i] -= attr["lr"] / np.sqrt(self.z[i] + attr["eps"]) * gr[i]

    def adam(self,xs,ys,attr):

        gr = [0,0]
        for i, dx in enumerate(self.derivatives):

            _, gr[i] = dx(xs,ys)

            self.z[i] = attr["b"][0] * self.z[i] + (1 - attr["b"][0]) * gr[i]
            self.o[i] = attr["b"][1] * self.o[i] + (1 - attr["b"][1]) * (gr[i]**2)

            #Bias correction
            z_corr = self.z[i] / (1 - (attr["b"][0]**self.t))
            o_corr = self.o[i] / (1 - (attr["b"][1]**self.t))


            #Update parameters
            self.theta[i] -= attr["lr"] * z_corr / (np.sqrt(o_corr) + attr["eps"])

        self.t += 1

    def wame(self,xs,ys,attr):

        gr = [0,0]
        for i, dx in enumerate(self.derivatives):
            _, gr[i] = dx(xs,ys)

            if gr[i] * self.grads_prev[i] > 0:
                self.step[i] = min(self.step[i] * attr["inc"], attr["step_sizes"][1])

            elif gr[i] * self.grads_prev[i] < 0:
                self.step[i] = max(self.step[i] * attr["dec"], attr["step_sizes"][0])

            self.z[i] = attr["a"] * self.z[i] + (1 - attr["a"] ) * self.step[i]
            self.o[i] = attr["a"] * self.o[i] + (1 - attr["a"]) * (gr[i]**2)

            self.theta[i] -= attr["lr"] * gr[i] / (self.o[i]) * self.z[i]

        self.grads_prev = gr


    def __str__(self):
        return f"f(yhat) = {round(self.theta[0][0],2)} + {round(self.theta[1][0],2)}x"


# Helper functions


def sample(X,Y,batch_size):
    idx = np.random.randint(0,len(X), batch_size)
    x = np.take(X, idx)
    y = np.take(Y, idx)
    return x,y

def evl(X, y, model):
    cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

def sample_batch(data,batch_size):
    st_batch = data.shape[0] // batch_size
    idx_end = st_batch * batch_size
    batch = np.split(data[:idx_end], st_batch)

    if data.shape[0] % batch_size != 0:
        batch += [data[idx_end:]]

    return batch

def visualise_board(log, title):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12,12))

    fig.suptitle(title, fontsize=16)
    ax1.plot(log["epochs"],log["T Loss"], c="tab:blue")
    ax1.xaxis.grid(True,linestyle=":",color='black')
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Train. Loss")

    ax2.plot(log["epochs"],log["T Accuracy"], c="tab:red")
    ax2.xaxis.grid(True,linestyle=":",color='black')
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Train. Accuracy")

    ax3.plot(log["epochs"],log["V Loss"], c="tab:green")
    ax3.xaxis.grid(True,linestyle=":",color='black')
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Val. Loss")

    ax4.plot(log["epochs"],log["V Accuracy"], c="tab:orange")
    ax4.xaxis.grid(True,linestyle=":",color='black')
    ax4.set_xlabel("Iterations")
    ax4.set_ylabel("Val. Accuracy")

def pairPlot(data):
    seaborn.pairplot(data,hue="income",plot_kws={"s": 3},dropna=True)

def corrMap(data):
    fig,ax = pyplot.subplots(figsize=(15,8))
    seaborn.heatmap(ax=ax,data=data.corr().round(2),annot=True,cmap=seaborn.diverging_palette(220,20),linewidth=2)

def progress_plot(ax,model,X,x):
    minX = min(X)
    maxX = max(X)
    ax.plot([minX,maxX], [model.yhat(minX),model.yhat(maxX)], c="tab:red", alpha=0.35, linewidth=0.1)
    ax.scatter(x,model.yhat(x), c="tab:red", alpha=0.35, linewidth=0.1)
