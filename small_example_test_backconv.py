# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:12:52 2019

@author: d2gu53
"""

import numpy as np
import os
import sys

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
label = 0

conv = Conv3x3(3)  # 28x28x1 -> 26x26x8
pool = MaxPool2()  # 26x26x8 -> 13x13x8
softmax = Softmax(4 * 3 * 3, 2)  # 13x13x8 -> 10

Conv3x3(3).filters
num_filters = 3
np.random.seed(seed=666); filter_conv = np.random.randn(num_filters, 3, 3) * 3# / 9
filter_conv = np.round(filter_conv)

###############################################################################
## pass in net from blog and own net
###############################################################################

########################### feedforward 

# feedforward conv layer
out_conv = conv.forward(image)
out_convown, filter_conv, inter = fun.convolute(image, filter_conv)
# feedforward maxpool layer
out_max = pool.forward(out_conv)
out_maxown, index_maxown = fun.max_pool(out_convown)
# feedforward softmax layer
out_soft, weights = softmax.forward(out_max) #
np.random.seed(seed=666); weight_soft = (np.random.randn(size, 2) / size) * 10
bias_soft = np.zeros(2)
probabilities, inter_soft = fun.softmax(output_maxpool=maxpool_map, 
                                        weight_matrix=weight_soft, 
                                        bias_vector=bias_soft)


##################### backprop

# backprop softmax
gradient_L = np.zeros(10)
gradient_L[label] = -1 / out_soft[label]
gradient_soft = softmax.backprop(gradient_L, 0.01)[0]
weight_soft, bias_soft, inter_softback, gradient_softown, inter_soft = fun.backprop_softmax(inter_soft=inter_soft, 
                     maxpool_shape=maxpool_map.shape,
                     probabilities=probabilities,
                     label=label, 
                     learn_rate=0.01)

# backprop maxpool
gradient_max = pool.backprop(gradient_soft)
feature_gradients = fun.backprop_maxpool(out_convown, index_maxown, gradient_softown)

feature_gradients.shape
# backprop conv
## for first and third filter outputs are the same, for second different?
gradient_conv = conv.backprop(gradient_max, learn_rate=0) 


feature_gradients[0, 0, 3]
np.prod(out_maxown.shape)
len(gradient_softown)
out_convown.shape
feature_gradients.shape

def backprop_conv(image, filter_conv, index_max, feature_gradient):           
    #gradient = gradient.reshape(shape_outmax)
    dpool_dfilter = np.zeros(shape=filter_conv.shape)
    n_filters = dpool_dfilter.shape[0]
    n_rows = dpool_dfilter.shape[1]
    n_cols = dpool_dfilter.shape[2]
    
    for f in range(n_filters):
        row_max, col_max = np.where(index_max[f] == True)
        for i in range(n_rows):
            for j in range(n_cols):
                for m, n in zip(row_max, col_max):
                    dpool_dfilter[f, i, j] += image[m+i, n+j] * feature_gradient[f, m, n]

    return dpool_dfilter      



backprop_conv(image, filter_conv, index_maxown, feature_gradients)

index_maxown[0]