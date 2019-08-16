# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:12:52 2019

@author: d2gu53
"""

import numpy as np
import os
import sys
import mnist
path = "C:/Users/D2GU53/Documents/master_arbeit/nn_in_r/cnn-from-scratch"
path2 = "C:/Users/D2GU53/Documents/master_arbeit/nn_in_r"
sys.path.append(path)
sys.path.append(path2)
import functions as fun
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax


np.random.seed(seed=666); image = np.random.randn(10, 8) * 3
image = np.round(image)
label = 5
test_images = mnist.test_images()[:2000]
test_labels = mnist.test_labels()[:2000]

image = test_images[0]
label = test_labels[0]

conv = Conv3x3(8)  # 28x28x1 -> 26x26x8
pool = MaxPool2()  # 26x26x8 -> 13x13x8
dim_maxpool = np.prod(13 * 13 * 8)
softmax = Softmax(dim_maxpool, 10)  # 13x13x8 -> 10

num_filters = 8
np.random.seed(seed=666); filter_conv = np.random.randn(num_filters, 3, 3) * 3# / 9
filter_conv = np.round(filter_conv)

###############################################################################
## pass in net from blog and own net
###############################################################################

########################### feedforward 

# feedforward conv layer
out_conv = conv.forward(image)
out_convown, filter_conv, inter = fun.convolute(image, filter_conv)

if np.sum(out_conv == out_convown) == np.prod(out_convown.shape):
    print("Yeaaaah!")
# feedforward maxpool layer

out_max = pool.forward(out_conv)
out_maxown = fun.max_pool(out_convown)

if np.sum(out_max== out_maxown) == np.prod(out_maxown.shape):
    print("Yeaaaah!")

# feedforward softmax layer
out_soft, weights, summe = softmax.forward(out_max) #
np.random.seed(seed=666); weight_soft = (np.random.randn(dim_maxpool, 10) / dim_maxpool) * 10
#bias_soft = np.zeros(6)
bias_soft = np.array([-5, -1, 4, 5, 6, 7] )
probabilities, inter_soft = fun.softmax(output_maxpool=out_max, 
                                        weight_matrix=weight_soft, 
                                        bias_vector=bias_soft)

if np.sum(out_soft== probabilities) == np.prod(out_soft.shape):
    print("Yeaaaah!")

##################### backprop

# backprop softmax
gradient_L = np.zeros(10)
gradient_L[label] = -1 / out_soft[label]
gradient_soft, dL_dw, weights_updated, biases_updated = softmax.backprop(gradient_L, 0.1)
weights_updatedown, biases_updatedown, gradient_softown = fun.backprop_softmax(inter_soft=inter_soft, 
#                     maxpool_shape=out_maxown.shape,
                     probabilities=probabilities,
                     label=label, 
                     learn_rate=0.1)

if np.sum(weights_updated == weights_updatedown) == np.prod(weights_updated.shape):
    print("Yeaaaah!")

if np.sum(biases_updated == biases_updatedown) == np.prod(biases_updated.shape):
    print("Yeaaaah!")

if np.sum(gradient_soft == gradient_softown) == np.prod(gradient_soft.shape):
    print("Yeaaaah!")



# so gradients are the same, but they do not get copied to correct entries in 
# feature map

# backprop maxpool
gradient_max = pool.backprop(gradient_soft)
#gradient_test = np.ones(shape=gradient_softown.shape)
gradient_maxown = fun.backprop_maxpool(out_convown, gradient_softown)

if np.sum(gradient_max == gradient_maxown) == np.prod(gradient_max.shape):
    print("Yeaaaah!")

gradient_conv, filter_update = conv.backprop(gradient_max, 0.01)
filter_updateown,gradient_convown = fun.backprop_conv(image, filter_conv ,
                                     feature_gradient=gradient_maxown, learn_rate=0.01)

if np.sum(gradient_conv == gradient_convown) == np.prod(gradient_conv.shape):
    print("Yeaaaah!")

if np.sum(filter_update == filter_updateown) == np.prod(filter_update.shape):
    print("Yeaaaah!")



