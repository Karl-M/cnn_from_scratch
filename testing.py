# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:55:43 2019

@author: d2gu53
"""

# testing stuff 

import numpy as np
import sys
import os
path = "C:/Users/D2GU53/Documents/master_arbeit/nn_in_r"
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


test_image = np.round(np.random.randn(6, 6) * 3)



test_image.shape
test_filter1 = np.array([[0, 0, 0], [1, 2, 1], [0, 0, 0]])
test_filter2 = test_filter1.T

test_filter = np.array([test_filter1, test_filter2])
test_conv, filter_mat, inter_conv = fun.convolute(image=test_image, filter_matrix=test_filter)


out_maxpool, index_maxpool = fun.max_pool(test_conv)

ind = np.where(index_maxpool[0] == True) # for these indices we need gradients with respect to inmage?


test_conv[index_maxpool]  # diese indices wollen wir mit deltaL * image[]



def backprop_maxpool(feature_map, index_max):
    
    feature_map_back = np.full(feature_map.shape, False)
    feature_map_back[index_max] =  True
    
    return feature_map_back


back_max = backprop_maxpool(feature_map=test_conv, index_max=index_maxpool)



backprop_conv(test_image, test_filter, back_max)



def backprop_max_conv(image, filter_conv, index_maxpool, learn_rate):
    
    n_filters, height, width = filter_conv.shape
    dConv = np.zeros(shape=(n_filters, height, width))
    
    for f in range(n_filters):
        row_index_max = np.where(index_maxpool[f] == True)[0]
        col_index_max = np.where(index_maxpool[f] == True)[1]
        print(row_index_max)
        print(col_index_max)
        for i in range(height):
            for j in range(width):
                test = 0
                for m in range(index_maxpool[f].shape[0]):
                    for n in range(index_maxpool[f].shape[1]):
                        if m in row_index_max and n in col_index_max:                
                            print(f"i: {i}, j: {j}, m: {m}, n: {n}, pixelwert: {image[i+m, j+n]}")
                            test += image[i+m, j+n] 
                            dConv[f, i, j] = test
                            
  #  filter_conv -= learn_rate * dConv
    
    return dConv


def backprop_max_conv(image, filter_conv, index_maxpool, learn_rate):
    
    n_filters, height, width = filter_conv.shape
    dConv = np.zeros(shape=(n_filters, height, width))
    
    for f in range(n_filters):
        row_index_max = np.where(index_maxpool[f] == True)[0]
        col_index_max = np.where(index_maxpool[f] == True)[1]
        print(row_index_max)
        print(col_index_max)
        for i in range(height):
            for j in range(width):
                test = 0
                for m in range(index_maxpool[f].shape[0]):
                    for n in range(index_maxpool[f].shape[1]):
                        if m == row_index_max[m] and n == col_index_max[n]:                
                            print(f"i: {i}, j: {j}, m: {m}, n: {n}, pixelwert: {image[i+m, j+n]}")
                            test += image[i+m, j+n] 
                            dConv[f, i, j] = test
                            
  #  filter_conv -= learn_rate * dConv
    
    return dConv  

  
index_maxpool[0].shape[1]

np.where(index_maxpool[0] == True)
backprop_max_conv(test_image, test_filter, index_maxpool, learn_rate=0.1)

test_filter

test_filter - 0.1 * back_out_conv

