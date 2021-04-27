import pytest
from NeuralNetwork import *

def test_forward():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    nn_test = NeuralNetwork(input_size = 2, hidden_layer_size = 3, output_size = 1)
    nn_test.params["W1"] = np.array([[0.1,0.6],[0.2,0.4],[0.3,0.7]])
    nn_test.params["W2"] = np.array([[0.1,0.4,0.9]])

    assert np.round(nn_test.forward(X),3).tolist()  == [[0.668, 0.712, 0.688, 0.728]]

def test_backward():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    nn_test = NeuralNetwork(input_size = 2, hidden_layer_size = 3, output_size = 1)
    nn_test.params["W1"] = np.array([[0.1,0.6],[0.2,0.4],[0.3,0.7]])
    nn_test.params["W2"] = np.array([[0.1,0.4,0.9]])
    nn_test.forward(X)
    dW2, db2, dW1, db1 = nn_test.backward(y,nn_test.d_mse)

    assert np.round(dW2,3).tolist() == [[0.024, 0.024, 0.025]]
    assert np.round(db2,3).tolist() == [[0.042]]
    assert np.round(dW1,4).tolist() ==  [[0.0004, 0.0005],[0.0016, 0.0019],[0.0027, 0.0034]]
    assert np.round(db1,5).tolist() ==  [[0.00097],[0.00392],[0.00808]]

def test_b_cross_entropy():
    nn_test = NeuralNetwork(input_size = 2, hidden_layer_size = 3, output_size = 1)
    y = np.array([[0, 1, 1, 0]])
    nn_test.A3 = np.array([[0.59, 0.2, 0.5, 0.7]])

    ce, loss = nn_test.b_cross_entropy(y, nn_test.A3)


    assert np.round(ce,2).tolist() == [[0.89, 1.61, 0.69, 1.20]]


def test_db_cross_entropy():
    nn_test = NeuralNetwork(input_size = 2, hidden_layer_size = 3, output_size = 1)
    y = np.array([[0, 1, 1, 0]])
    nn_test.A3 = np.array([[0.59, 0.2, 0.5, 0.7]])

    assert np.round(nn_test.db_cross_entropy(y),2).tolist() == [[2.44, -5.00, -2.00, 3.33]]


def test_d_mse():
    nn_test = NeuralNetwork(input_size = 2, hidden_layer_size = 3, output_size = 1)
    y = np.array([[0, 1, 1, 0]])
    nn_test.A3 = np.array([[0.59, 0.2, 0.5, 0.7]])

    assert np.round(nn_test.d_mse(y),2).tolist() == [[0.15, -0.20, -0.12, 0.18]]


def test_gd():
    nn_test = NeuralNetwork(input_size = 2, hidden_layer_size = 2, output_size = 1)
    nn_test.params["W1"] = np.array([[1.2, 0.2],[0.6, 1.5]])
    nn_test.grads["W1"] = np.array([[ 1.3, -0.6],[ 0.6, -0.1]])

    nn_test.gd(attr={"lr":0.1})
    assert np.round(nn_test.params["W1"],2).tolist() == [[1.07, 0.26],[0.54, 1.51]]

def gd_m():
    pass

def test_rprop():
    nn_test = NeuralNetwork(input_size = 2, hidden_layer_size = 2, output_size = 1)
    nn_test.params["W1"] = np.array([[1.2, 0.2],[0.6, 1.5]])
    nn_test.grads["W1"] = np.array([[ 1.3, -0.6],[ 0.6, -0.1]])
    nn_test.grads_prev["W1"] = np.array([[-1.07, 0.26], [0.54, 1.51]])
    nn_test.step["W1"] = np.array([[0.05, 0.14], [0.63, 1.25]])

    nn_test.rprop(attr={"inc":1.2, "dec":0.5, "step_sizes":(0.0001,50)})

    assert np.round(nn_test.params["W1"],2).tolist() == [[1.18, 0.27],[-0.16, 2.12]]

def rmsprop():
    nn_test = NeuralNetwork(input_size = 2, hidden_layer_size = 2, output_size = 1)
    nn_test.params["W1"] = np.array([[1.2, 0.2],[0.6, 1.5]])
    nn_test.grads["W1"] = np.array([[ 1.3, -0.6],[ 0.6, -0.1]])
    nn_test.z["W1"] = np.array([[-1.07, 0.26], [0.54, 1.51]])
    nn_test.step["W1"] = np.array([[0.05, 0.14], [0.63, 1.25]])

def adam():
    nn_test = NeuralNetwork(input_size = 2, hidden_layer_size = 2, output_size = 1)
    nn_test.params["W1"] = np.array([[1.2, 0.2],[0.6, 1.5]])
    nn_test.grads["W1"] = np.array([[ 1.3, -0.6],[ 0.6, -0.1]])
    nn_test.z["W1"] = np.array([[-1.07, 0.26], [0.54, 1.51]])
    nn_test.o["W1"] = np.array([[-0.70, 0.62], [0.45, 0.15]])
    nn_test.step["W1"] = np.array([[0.05, 0.14], [0.63, 1.25]])

    nn_test.adam(t=10, lr=0.01, b=(0.9,0.999), eps=0.0001)

def test_wame():
    nn_test = NeuralNetwork(input_size = 2, hidden_layer_size = 2, output_size = 1)
    nn_test.params["W1"] = np.array([[1.2, 0.2],[0.6, 1.5]])
    nn_test.grads["W1"] = np.array([[ 1.3, -0.6],[ 0.6, -0.1]])
    nn_test.grads_prev["W1"] = np.array([[-1.07, 0.26], [0.54, 1.51]])
    nn_test.z["W1"] = np.array([[-1.07, 0.26], [0.54, 1.51]])
    nn_test.o["W1"] = np.array([[-0.70, 0.62], [0.45, 0.15]])
    nn_test.step["W1"] = np.array([[0.05, 0.14], [0.63, 1.25]])

    nn_test.wame(attr={"lr":0.1, "inc":1.2, "dec":0.1,"a":0.9,"step_sizes":(0.01,100)})

    assert np.round(nn_test.params["W1"],2).tolist() == [[0.91, 0.63],[0.36, 1.55]]
