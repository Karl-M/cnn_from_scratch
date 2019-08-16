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


train.training(2, 10, 8, 
    test_images, test_labels
    , learn_rate=0.1, 
    print_acc=True,
    weights_conv=None,
    weights_soft=None)[3]


# run model from blog with same data
conv = Conv3x3(8)  # 28x28x1 -> 26x26x8
pool = MaxPool2()  # 26x26x8 -> 13x13x8
dim_maxpool = np.prod(13 * 13 * 8)
softmax = Softmax(dim_maxpool, 10)


conv.filters
softmax.weights

for i in range(2):
    print("this is iteration", i)
    image = test_images[i] / 255 -0.5
    label = test_labels[i]
    out_conv = conv.forward(image)
    out_max = pool.forward(out_conv)
    out_soft, weights, summe = softmax.forward(out_max) #
    print(out_soft)
    gradient_L = np.zeros(10)
    gradient_L[label] = -1 / out_soft[label]
    gradient_soft, dL_dw, weights_updated, biases_updated = softmax.backprop(gradient_L, 0.1)
    gradient_max = pool.backprop(gradient_soft)
    gradient_conv, filter_update = conv.backprop(gradient_max, 0.1)
    







conv, soft = fun.training(1001, 10, 8, test_images, test_labels, 
                          weights_conv=conv,
                          weights_soft=soft,
                          learn_rate=0.01, print_acc=True)


good_conv, good_soft = conv, soft


conv["bias_vector"]
soft["bias_vector"]

# das ganze mal für ein Bild testen?


fun.convolute(test_images[0], filter_matrix_conv)

it_regions = Conv3x3.iterate_regions(8, image=test_images[0])
Conv3x3.forward(it_regions, test_images[0])



5 * np.array([1, 3, 4])



feature_back[0].shape



