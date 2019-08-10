# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:55:43 2019

@author: d2gu53
"""

# testing stuff 

import numpy as np

path = "/home/konstantin/Documents/master_arbeit/cnn_from_scratch"
#path = "C:/Users/D2GU53/Documents/master_arbeit/nn_in_r"
sys.path.append(path)
import functions as fun


def max_pool(feature_map):
    
    if len(feature_map.shape) < 3:
        number_filters = 1
        height, width = feature_map.shape
        feature_map = np.reshape(test_mat, (1, height, width))
    else: 
        number_filters, height, width = feature_map.shape
        
    pooling_map = np.zeros(shape=(number_filters, height // 2, width // 2))
    
    # need indices from max for backprop
    index = np.full(feature_map.shape, False) # index array
    
    for k in range(number_filters):
       for i in range(height // 2):
            for j in range(width // 2):
                res = feature_map[k, i*2:i*2 + 2, j*2:(j*2 + 2)]
                pooling_map[k, i, j] = np.amax(res)                
                where = np.where(res == np.max(res))
                #print(where[0][0], where[1][0])
                index[k, i*2:i*2 + 2, j*2:(j*2 + 2)][where[0][0], where[1][0]]  = 1
              #  print(index)
                
    return pooling_map, index

test_mat = np.random.randn(12, 4)
test_mat = np.round(test_mat, 2)
pooling_map, index_maxpool = max_pool(feature_map=test_mat)

test_mat[index_maxpool[0, :, :]]


def backprop_maxpool(feature_map, index_max, gradient):
    #print(output_maxpool.shape)
    height, width = output_maxpool.shape[1:]
  #  mat_in_maxpool = np.zeros(shape=(2*height, 2*width))
    mat_in_maxpool = test
    print(height)
    print(width)
    print(mat_in_maxpool)
    
    for j in range(width):
        print(f"spalte {j}: {mat_in_maxpool[:, j:j+3]}")
#    for i in range(height):
#        print(f"reihe {i}: {mat_in_maxpool[i, :]}")# = output_maxpool[i, j]
    return 
        
backprop_maxpool(max_pool(feature_map=test_mat), test_mat)        






test_image = np.array([list(range(1, 7)), 
          list(range(6, 12)),
          list(range(11, 17)),
          list(range(16, 22)),
          list(range(21, 27)),
          list(range(26, 32))])

test_image.shape
test_filter1 = np.array([[0, 0, 0], [1, 2, 1], [0, 0, 0]])
test_filter2 = test_filter1.T

test_filter = np.array([test_filter1, test_filter2])
bias_vector = np.array([0, 0])
bias_vector[1]
len(test_filter.shape)
test_conv, inter_conv = fun.convolute(image=test_image, filter_matrix=test_filter, bias_vector=bias_vector)

out_maxpool, index_maxpool = fun.max_pool(test_conv)

out_maxpool