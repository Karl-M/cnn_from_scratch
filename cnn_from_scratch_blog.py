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
path = "/home/konstantin/Documents/master_arbeit/cnn_from_scratch"
#path = "C:/Users/D2GU53/Documents/master_arbeit/nn_in_r"
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


conv, soft = fun.training(1001, 10, 8, test_images, test_labels, learn_rate=0.01, print_acc=True)

conv, soft = fun.training(1001, 10, 8, test_images, test_labels, 
                          weights_conv=conv,
                          weights_soft=soft,
                          learn_rate=0.01, print_acc=True)


good_conv, good_soft = conv, soft

conv["bias_vector"]
soft["bias_vector"]

