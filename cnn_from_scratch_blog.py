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
import test_cnn as test

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

# these two approches seem to deliver different results, although the output of these
# is exaclty the same in small_example_test_backconv.py
# trying to compare every result on its own

test.debug_cnn(2)




