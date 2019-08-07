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


# why use softmax? To quantify how sure we are of our prediction
# and use for example cross-entropy-loss

# define forward pass through network

test_images = mnist.test_images()[20:2000]
test_labels = mnist.test_labels()[20:2000]

n_classes = 10
input_len = 1014
number_filters = 6
weight_matrix = np.random.randn(input_len, n_classes) / (input_len)
weight_matrix_old  = weight_matrix
bias_vector = np.random.randn(n_classes) / (n_classes)
bias_vector_old = bias_vector

filter_matrix_conv = np.random.randn(number_filters, 3, 3) / 9
bias_vector_conv = np.random.randn(number_filters) / number_filters

filter_matrix_conv.shape[0]    
weight_matrix.shape[1]
#bias_vector = bias_vector_old


def training(n_iter, n_classes, n_filter, training_data, label, 
             learn_rate=0.01, print_acc=True):
    
    num_correct = 0
    input_dim = int((((training_data[0].shape[0] - 3 + 1) / 2) ** 2) * n_filter)
    filter_matrix_conv = np.random.randn(n_filter, 3, 3) / 9
    bias_vector_conv = np.random.randn(n_filter) / n_filter
    
    weight_matrix_soft = np.random.randn(input_dim, n_classes) / (input_dim)
    bias_vector_soft = np.random.randn(n_classes) / (n_classes)
    
    for i in range(n_iter):
        image = training_data[i] / 255 - 0.5
        
        out_conv, intermediates_conv = fun.convolute(
                image=image, 
                filter_matrix=filter_matrix_conv,
                bias_vector=bias_vector_conv)
        
        out_maxpool = fun.max_pool(feature_map=out_conv)
        
        probabilities, intermediates_soft = fun.softmax(
                output_maxpool=out_maxpool, 
                weight_matrix=weight_matrix_soft,
                bias_vector=bias_vector_soft)
        
        weight_matrix_soft, bias_vector_soft = fun.backprop(
                probabilities=probabilities,
                inter_soft=intermediates_soft,
                label=label[i],
                learn_rate=learn_rate)
        
            
  #      loss = -np.log(probabilities[label])
        prediction = np.argmax(probabilities)
     #   print(f"prediction: {prediction}")
     #   print(f"label: {label[i]}")
        acc = 1 if prediction == label[i] else 0
        
        num_correct += acc
        
        if i % 100 == 0 and i != 0 and print_acc:
            accuracy = num_correct / i
            print(f"accuracy for the first {i} samples: {accuracy}")
            print(f"{num_correct} predictions for {i} samples were correct")

    return

training(1001, 10, 6, test_images, test_labels, learn_rate=0.1, print_acc=True)



