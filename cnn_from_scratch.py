# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:39:36 2019

@author: d2gu53
"""

import numpy as np
import sys
import os
sys.path.append("C:\\Users\D2GU53\Documents\master_arbeit\cats_and_dogs")
import functions as fun


# generate matrix representations of images and run NN over it

X = np.zeros(shape=(9, 9))

sample_horizontal = []
for i in range(1000):
    X = np.zeros(shape=(9, 9))
    a = np.random.choice(range(0, 9))
    b = np.random.choice(range(0, 9))
    X[a, ] = 1
    X[b, ] = 2
    sample_horizontal.append(X)

sample_vertical = []
for i in range(1000):
    X = np.zeros(shape=(9, 9))
    a = np.random.choice(range(0, 9))
    b = np.random.choice(range(0, 9))
    X[: , a] = 1
    X[: , b] = 2
    sample_vertical.append(X)



data = [sample_horizontal, sample_vertical]

labels = []
labels.append([np.repeat(1, 1000)]) # horiontal edges get label one
labels.append([np.repeat(0, 1000)]) # vertical edges get label zero


type(data[1][2])
# build 3x3 filter

filter_mat_ver = np.zeros(shape = (3, 3))
filter_mat_ver[:, 1] = 1

filter_mat_hor = np.zeros(shape = (3, 3))
filter_mat_hor [1, :] = 1


result = []
for i in range(7):
    for j in range(7):
        res = data[1][0][i:i+3, j:j+3]
        result.append(res)

filter_mat.shape
data[1][0].shape[0]

def convolute(image, filter_mat):
    
    width = image.shape[0]
    height = image.shape[1]
    filter_size = filter_mat.shape[0]
    feature_map = np.zeros(shape=(width - filter_size + 1, height - filter_size + 1))
    for i in range(width - filter_size+1):
        for j in range(height - filter_size+1):
            res = image[i:i+filter_size, j:j+filter_size] * filter_mat
            feature_map[i, j] = np.sum(res)
    #result = [re]
    
    return feature_map
            
W1 = filter_mat_hor
W2 = filter_mat_ver
b1 = 1
b2 = 2    
model = {"W1": W1, "W2": W2, "b1": b1, "b2": b2}

fun.feed_forward( )



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










