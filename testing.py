# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:55:43 2019

@author: d2gu53
"""

# testing stuff 

import numpy as np

def max_pool(feature_map):
    
    if len(feature_map.shape) < 3:
        number_filters = 1
        height, width = feature_map.shape
        feature_map = np.reshape(test_mat, (1, height, width))
    else: 
        number_filters, height, width = feature_map.shape
        
    pooling_map = np.zeros(shape=(number_filters, height // 2, width // 2))
    
    # need indices from max for backprop
    index = np.zeros(shape=pooling_map.shape) # index array
    
    for k in range(number_filters):
       for i in range(height // 2):
            for j in range(width // 2):
                res = feature_map[k, i*2:i*2 + 2, j*2:(j*2 + 2)]
                pooling_map[k, i, j] = np.amax(res)                
                where = np.where(a == np.max(a))
                index[k, i, ] = where[0][0], where[1][0]

                
    return pooling_map


test_mat = np.random.randn(12, 4)
test_mat = np.round(test_mat, 2)

np.max(test_mat[0:3, 0:3])
np.argmax(test_mat[0:2, 0:2], axis=0)

help(np.argmax)
len(test_mat.shape)

1, test_mat.shape

a = np.array([[1,2,3],[4,3,5]])
b = np.zeros(shape=a.shape)
a

help(np.argmax)
np.argmax(a, out=b)
np.reshape(test_mat, (1, 12, 4)).shape

test_mat.shape

result = np.where(a == np.max(a))
index = result[0][0], result[1][0]

np.where(a == np.max(a))[0]
a[index]

max_pool(feature_map=test_mat).shape[1:]

np.am

def backprop_maxpool(output_maxpool, test):
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