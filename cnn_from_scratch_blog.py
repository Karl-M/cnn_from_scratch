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

#cats_and_dogs_folder = Path("C:\\Users\D2GU53\Documents\master_arbeit\cats_and_dogs")

#if not cats_and_dogs_folder.is_file():
#    raise AssertionError(f"Wrong folder: {cats_and_dogs_folder}")
#sys.path.append(cats_and_dogs_folder)


# why use softmax? To quantify how sure we are of our prediction
# and use for example cross-entropy-loss

# define forward pass through network

test_images = mnist.test_images()[:2000]
test_labels = mnist.test_labels()[:2000]


conv, soft, feature_back = fun.training(101, 10, 8, test_images, test_labels, learn_rate=0.01, print_acc=True)

conv, soft = fun.training(1001, 10, 8, test_images, test_labels, 
                          weights_conv=conv,
                          weights_soft=soft,
                          learn_rate=0.01, print_acc=True)


good_conv, good_soft = conv, soft

conv["bias_vector"]
soft["bias_vector"]
n_filter = 8
filter_matrix_conv = np.random.randn(n_filter, 3, 3) / 9
bias_vector_conv = np.random.randn(n_filter) / n_filter

test_conv, intermediates = fun.convolute(test_images[0], filter_matrix_conv, bias_vector_conv)


pooling_map, index = fun.max_pool(test_conv)

test_conv[0].shape
index.shape
index[0, 0:4, 0:4]
test_labels[0]
delta = np.random.randn(10)
delta = np.round(delta)
delta[7] = 50

# funktion überschreibt test_conv? wegen immutable objects????
feature_back = fun.backprop_maxpool(test_conv[0, :, :], index[0, :, :], delta, 7)

test_conv[0]


np.sum(test_conv[0] == feature_back, axis=0)
feature_back.shape
test_conv[0, 0:4, 0:4]
feature_back[0:4, 0:4]



index.shape
test_conv.shape
feature_back.shape
test_conv[0, 10:12, 10:12]

feature_back[0, 10:12, 10:12]
pooling_map.shape
conv, soft, feature_back = fun.training(1, 10, 8, test_images, test_labels, learn_rate=0.01, print_acc=True)













