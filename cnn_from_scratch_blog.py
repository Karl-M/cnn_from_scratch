# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:26:32 2019

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
from pathlib import Path
import functions as fun
import mnist

cats_and_dogs_folder = Path("C:\\Users\D2GU53\Documents\master_arbeit\cats_and_dogs")

if not cats_and_dogs_folder.is_file():
    raise AssertionError(f"Wrong folder: {cats_and_dogs_folder}")
sys.path.append(cats_and_dogs_folder)


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

# convolution operation with randomnly initialized filters
def convolute(image, number_filters):
    
    filter_matrix = np.random.randn(number_filters, 3, 3)    
    width, height = image.shape
    feature_map = np.zeros(shape=(number_filters, width - 3 + 1, height - 3 + 1))

    for k in range(number_filters):
       for i in range(width - 3 + 1):
            for j in range(height - 3 + 1):
                res = image[i:(i + 3), j:(j + 3)] * filter_matrix[k]
                feature_map[k, i, j] = np.sum(res)
    
    return feature_map


test_features = convolute(data[0][1], 5)
test_features.shape

def max_pool(feature_map):
    number_filters, width, height = feature_map.shape
#    print(feature_map.shape)
    pooling_map = np.zeros(shape=(number_filters, width // 2, height // 2))
    print(pooling_map.shape)
    for k in range(number_filters):
       for i in range(int(width / 2)):
            for j in range(int(height / 2)):
                print(f"k={k} i={i} j={j}")
                res = feature_map[k, i*2:i*2 + 2, j*2:(j*2 + 2)]
                print(res)
                print(np.max(res))
                print(f"shape = {res.shape}")
                pooling_map[k, i, j] = np.max(res)
                
    return pooling_map

max_pool(test_features).shape
    
test_features.shape

max_pool(test_features)