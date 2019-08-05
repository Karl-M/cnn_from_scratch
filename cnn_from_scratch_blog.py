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

path = "/home/konstantin/Documents/master_arbeit/cnn_from_scratch"
sys.path.append(path)
import functions as fun

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

test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

def feed_forward(image, label, number_filters, n_classes):
    
    image = image / 255    
    out_conv = fun.convolute(image=image, number_filters=number_filters)
    out_maxpool = fun.max_pool(feature_map=out_conv)
    out_softmax = fun.softmax(output_maxpool=out_maxpool, n_classes=n_classes)
    
    # compute cross entropy loss. Normaly cross entropy involves summing
    # over all classes and predictions but since the true probability is 
    # either 1 or 0 and 1 only once for every image, we dont need to sum
    # over all classes
    
    loss = -np.log(out_softmax[label])
    acc = 1 if np.max(out_softmax == label) else 0
    
    return out_softmax, loss, acc, label


for i in range(100):
    probabilities, loss, acc, label = feed_forward(test_images[i], test_labels[i], 
                                 number_filters=6, n_classes=10)
    print(loss, acc, label)





