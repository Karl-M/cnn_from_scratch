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
#path2 = "/home/konstantin/Documents/master_arbeit/cnn-from-scratch"
path = "C:/Users/D2GU53/Documents/master_arbeit/nn_in_r"
path2 = "C:/Users/D2GU53/Documents/master_arbeit/nn_in_r/cnn-from-scratch"

sys.path.append(path)
sys.path.append(path2)

import functions as fun
import training as train

from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

#cats_and_dogs_folder = Path("C:\\Users\D2GU53\Documents\master_arbeit\cats_and_dogs")

#if not cats_and_dogs_folder.is_file():
#    raise AssertionError(f"Wrong folder: {cats_and_dogs_folder}")
#sys.path.append(cats_and_dogs_folder)


# why use softmax? To quantify how sure we are of our prediction
# and use for example cross-entropy-loss

# define forward pass through network

test_images = mnist.test_images()[:2000]
test_labels = mnist.test_labels()[:2000]
# these two approches seem to deliver different results, although the output of these
# is exaclty the same in small_example_test_backconv.py
# trying to compare every result on its own

num_filters = 8
np.random.seed(seed=666); filter_conv = np.random.randn(num_filters, 3, 3) * 3# / 9
filter_conv = np.round(filter_conv)

dim_maxpool = 13 * 13 *8
np.random.seed(seed=666); weight_soft = (np.random.randn(dim_maxpool, 10) / dim_maxpool) * 10
bias_soft = np.zeros(10)

for i in range(2):
    image = test_images[i] / 255 - 0.5
    label = test_labels[i]
    feature_map, filter_conv = fun.convolute(image=image, filter_matrix=filter_conv)
    maxpool_map = fun.maxpool(feature_map=feature_map)
    probs, inter_soft = fun.softmax(maxpool_map, 
                                    weight_matrix=weight_soft, 
                                    bias_vector=bias_soft)
    weight_soft, bias_soft, gradient_soft_own = fun.backprop_softmax(inter_soft=inter_soft,
                                                       probabilities=probs,
                                                       label = label,
                                                       learn_rate=0.1) 
    gradient_max_own = fun.backprop_maxpool(feature_map=feature_map, 
                                            gradient=gradient_soft_own)
    filter_conv = fun.backprop_conv(image=image, filter_conv=filter_conv,
                                    gradient=gradient_max_own, learn_rate=0.1)
 


# run model from blog with same data
conv = Conv3x3(8)  # 28x28x1 -> 26x26x8
pool = MaxPool2()  # 26x26x8 -> 13x13x8
dim_maxpool = np.prod(13 * 13 * 8)
softmax = Softmax(dim_maxpool, 10)

for i in range(2):
    print("this is iteration", i)
    image = test_images[i] / 255 -0.5
    label = test_labels[i]
    out_conv = conv.forward(image)
    #print(out_conv)   
    out_max = pool.forward(out_conv)
    out_soft, weights, summe = softmax.forward(out_max) #
#    print(out_soft)
    gradient_L = np.zeros(10)
    gradient_L[label] = -1 / out_soft[label]
    gradient_soft, dL_dw, weights_updated, biases_updated = softmax.backprop(gradient_L, 0.1)
    gradient_max = pool.backprop(gradient_soft)
    gradient_conv, filter_update = conv.backprop(gradient_max, 0.1)
    


################## compare feedforward ####################################
###########################################################################
if np.sum(feature_map == out_conv) == np.prod(feature_map.shape):
    print("Yeaaah!")
conv.filters == filter_conv # 
# after first iteration these are not the same anymore,
# since they get updated
if np.sum(maxpool_map == out_max) == np.prod(out_max.shape):
    print("Yeaaah!")
if np.sum(probs == out_soft) == np.prod(out_soft.shape):
    print("Yeaaah!")

######################### compare backprop #################################
############################################################################
 
######## softmax: gradients:    
if np.sum(gradient_soft == gradient_soft) == np.prod(gradient_soft.shape):
    print("Yeaaah!")
## weight updates weight matrix softmax layer
if np.sum(weight_soft == weights_updated) == np.prod(weight_soft.shape):
    print("Yeaaah!")
## weight updates bias vector
if np.sum(bias_soft == biases_updated) == np.prod(biases_updated.shape):
    print("Yeaaah!")
#### maxpool
if np.sum(gradient_max_own== gradient_max) == np.prod(gradient_max.shape):
    print("Yeaaah!")
## conv
if np.sum(filter_conv == filter_update) == np.prod(filter_conv.shape):
    print("Yeaaah!")
    
    
# So! After two runs the predicted probabilities are already different, why?




