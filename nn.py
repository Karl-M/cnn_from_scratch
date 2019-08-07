# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:11:30 2019

@author: d2gu53
"""

#### Neural Net in Python

import numpy as np
from sklearn import datasets
from sklearn import linear_model
import sklearn
from matplotlib import pyplot as plt
from src.Helper import Helper

helper = Helper()

np.random.seed(0)
X, y = datasets.make_moons(1000, noise=0.3)

clf = sklearn.linear_model.LogisticRegression()
clf.fit(X, y) ### überschreibt das clf Objekt

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
X = helper.plot_decision_boundary(lambda x: clf.predict(x), X)
plt.show()

### implement back propagation
num_examples = len(X) # size training set
nn_input_dim = 2 # input dimensionality
nn_output_dim = 2 # output dimensionality
h_dim = 5 # dimension of hidden layer
# Gradient descent parameters
nu = 0.01 # learning rate
reg_lambda = 0.01 # regularization strength

X : np.ndarray = X

### implement loss function
np.sum(X[0:3, :], axis=1)
len(np.sum(X, axis=1))

def calculate_loss(model):
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
    z1 = X.dot(W1) + b1
    a1 : np.ndarray = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2) # for softmax
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) #probabilities
    ## meine loss Berechnung funktioniert nicht, weil wir nicht mit den echten Ausprägungen
    # multiplizieren, sondern mit den Wahrscheinlichkeiten, dass Label zu erhalten (1 oder 0)
  #  data_loss_per_sample = -  (y * np.log(probs[:, 0]) + (1 - y) * np.log(probs[:, 1]))
  #  data_loss = np.sum(data_loss_per_sample)
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)
    return data_loss

def feed_forward(model, data):
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
    z1 = data.dot(W1) + b1
    a1 : np.ndarray = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2) # for softmax
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) #probabilities
    model = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    intermediates = {"z1": z1, "a1": a1, "z2": z2} 
    return probs, model, intermediates

W1 = np.random.randn(2, 2)

np.tanh(X.dot(W1) + (1, 2)).shape

def backprop(probs, model, intermediates, data):
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
    z1 = intermediates["z1"]
    a1 = intermediates["a1"]
    z2 = intermediates["z2"]
   # Backpropagation
    delta3 = probs
    delta3[range(num_examples), y] -= 1
    dW2 = (a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0)
    delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
    dW1 = np.dot(data.T, delta2)
    db1 = np.sum(delta2, axis=0)
    
    # update weights
    W1 += -nu * dW1
    b1 += -nu * db1
    W2 += -nu * dW2
    b2 += -nu * db2    

    model["W1"], model["b1"], model["W2"], model["b2"] = W1, b1, W2, b2
    return model


def training(n_iter, data, print_acc=True, print_loss=True, mini_batch_size=1):
    W1 = np.random.randn(nn_input_dim, h_dim)
    b1 = np.random.randn(h_dim)
    W2 = np.random.randn(h_dim, nn_output_dim)
    b2 = np.random.randn(nn_output_dim)
    
    model = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    #data = X    
    for iter in range(n_iter):
        probs, model, intermediates = feed_forward(model, X)
        ## stochastic gradient implementation
        if mini_batch_size == 1:
            model = backprop(probs, model, intermediates, data)
        else:
            batch_size = round(len(X) * mini_batch_size)
            batch = np.random.choice(range(len(X)), replace=False, size=batch_size)
            model = backprop(probs, model, intermediates, batch)    
        if print_acc and iter % 1000 == 0:
            predictions = np.argmax(feed_forward(model, X)[0], axis=1)
            print(f"accuracy bei iteration {iter}", np.mean(predictions == y))
        if print_loss and iter % 1000 == 0:
            print(f"loss bei iteration {iter}", calculate_loss(model))
            
    return model

test_run = training(1000, data=X, print_acc=False, print_loss=False)
test_run = training(10000, print_acc=True, print_loss=True, data=X, mini_batch_size=0.1)


predictions = predict(test_run, X)[0]

2000% 1000
predictions = np.argmax(feed_forward(test_run, X)[0], axis=1)
np.mean(predictions == y)



calculate_loss(test_run)
    
### implement mini batch

batch_size = 0.112

batch_size = round((len(X) -1) * batch_size)

batch = np.random.choice(range(len(X)), replace=False, size=batch_size)

(X[batch, :])
X.shape
type(X)






















