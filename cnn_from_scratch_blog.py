# -*- coding: utf-8 -*-
"""
#Created on Mon Aug  5 15:26:32 2019

@author: d2gu53
"""

# cnn from scratch in python 
# https://victorzhou.com/blog/intro-to-cnns-part-1/

# reasons to use cnn's for image classification

# normal NN would be to big
# the information apixel contains is only useful with respect to neighbouring pixels
# position of an object is irrelevant

# convolution helps us look for specific localized image features 
import numpy as np
import sys
#from pathlib import Path
import mnist
import os
#path = "/home/konstantin/Documents/master_arbeit/cnn_from_scratch"
path = "C:/Users/D2GU53/Documents/master_arbeit/nn_in_r"
sys.path.append(path)
import functions as fun

os.listdir(path)

#cats_and_dogs_folder = Path("C:\\Users\D2GU53\Documents\master_arbeit\cats_and_dogs")

#if not cats_and_dogs_folder.is_file():
#    raise AssertionError(f"Wrong folder: {cats_and_dogs_folder}")
#sys.path.append(cats_and_dogs_folder)


# generate some data
image_size = 10

sample_horizontal = []
for i in range(1000):
    X = np.zeros(shape=(image_size , image_size ))
    a = np.random.choice(range(0, image_size ))
    b = np.random.choice(range(0, image_size ))
    X[a, ] = 1
    X[b, ] = 2
    sample_horizontal.append(X)

sample_vertical = []
for i in range(1000):
    X = np.zeros(shape=(image_size , image_size ))
    a = np.random.choice(range(0, image_size ))
    b = np.random.choice(range(0, image_size ))
    X[: , a] = 1
    X[: , b] = 2
    sample_vertical.append(X)



data = [sample_horizontal, sample_vertical]

test_features = fun.convolute(data[0][1], 5)
test_features.shape


fun.max_pool(test_features).shape
    
test_features.shape

test_pool = fun.max_pool(test_features)


## build fully connected layer for predition
num_filter, height, width = test_pool.shape


n_classes = 10

weight_matrix = np.random.randn(num_filter * height * width, 10)




test_conv = fun.convolute(data[1][12], 6)
test_maxpool = fun.max_pool(test_conv)
test_softmax = fun.softmax(test_maxpool, 10)

test_softmax

# why use softmax? To quantify how sure we are of our prediction
# and use for example cross-entropy-loss

# define forward pass through network

test_images = mnist.test_images()[20:2000]
test_labels = mnist.test_labels()[20:2000]

import functions as fun

n_classes = 10
input_len = 1014
weight_matrix = np.random.randn(input_len, n_classes) / (input_len)
weight_matrix_old  = weight_matrix
bias_vector = np.random.randn(n_classes) / (n_classes)
bias_vector_old = bias_vector
#bias_vector = bias_vector_old

num_correct = 0
for i in range(1000):
    #false = 0
 #   print(f"bias_vector at iteration {i} : {bias_vector}")
  #  print(f"weight matrix at iteration {i}: {weight_matrix}")
    result = fun.feed_forward(image=test_images[i], 
                              label=test_labels[i], 
                                 number_filters=6, 
                                 n_classes=10,
                                 weight=weight_matrix ,
                                 bias=bias_vector,
                                 learn_rate=0.1)
    
    weight_matrix = result[6]
    bias_vector = result[7]
    update_vec = result[8]["dLoss"]
#    print(f'iteration: {i}: dLoss={result[8]["dLoss"]}')
#    print(f'iteration: {i}: dSoft={result[8]["dSoft"]}')
#    print(f'iteration: {i}: sum_exp={result[5]["sum_exp"]}')
#    print(f'iteration: {i}: exp={result[5]["exp"]}')
#    print(f'iteration: {i}: input_softmax={result[5]["input_softmax"]}')
#    print(f"Nans in Spalten bei i={i}: {np.sum(np.isnan(weight_matrix), axis=0)}")
#    print(f"Nans in Vector bei i={i}: {np.isnan(bias_vector)}")
#    print("")

    #print(f"weight updates für bias vector: {update_vec}")
  #  print(f"delta_L: {update_vec}")
    
    #print(f"Einträge im bias vector noch gleich? i = {i}: {bias_vector == bias_vector_old}")
    #print(f"Spalten der weight matrix noch gleich? i = {i}: {np.sum(weight_matrix == weight_matrix_old, axis=0)}")
    #print(np.sum(weight_matrix == weight_matrix_old, axis=1))
    #print(f"bias_vector at iteration {i} : {bias_vector}")    
    #print(f"weight matrix at iteration {i}: {weight_matrix}")
#    print(weight_matrix)
   # print(prediction, label, acc)
    num_correct += result[2]
   # print(num_correct)
    if i % 100 == 0 and i != 0:
        accuracy = num_correct / i
        print(f"accuracy for the first {i} samples: {accuracy}")
        print(f"{num_correct} predictions for {i} samples were correct")
  #      print(f"bias_vector at iteration {i} : {bias_vector}")
 #       print(f"weight matrix at iteration {i}: {weight_matrix}")
        print(f"Nans in Spalten bei i={i}: {np.sum(np.isnan(weight_matrix), axis=0)}")
        print(f"Nans in Vector bei i={i}: {np.isnan(bias_vector)}")
    
    #print(probabilities[label], prediction, label, acc)

weight_matrix
weight_matrix_old

np.sum(weight_matrix == weight_matrix_old, axis=1)
np.sum(np.isnan(weight_matrix), axis=0)
np.isnan(bias_vector)


bias_vector
bias_vector_old
result[6]
# 10% accuracy is equivalent to random guessing, to do better, we need to train
# the network. Training consists of two phases. A forward pass and a backward pass.

test_images[1].shape

np.prod(test_images.shape)



test_images[1] / 255 -0.5


