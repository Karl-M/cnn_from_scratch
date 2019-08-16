#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 19:07:31 2019

@author: konstantin
"""

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



test_images = mnist.test_images()[:2000]
test_labels = mnist.test_labels()[:2000]

import functions as fun
import training as train
import test_cnn as test

from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

def debug_cnn(n_iter):
    num_filters = 8
    np.random.seed(seed=666); filter_conv = np.random.randn(num_filters, 3, 3) * 3# / 9
    filter_conv = np.round(filter_conv)
    
    dim_maxpool = 13 * 13 *8
    np.random.seed(seed=666); weight_soft = (np.random.randn(dim_maxpool, 10) / dim_maxpool) * 10
    bias_soft = np.zeros(10)
    
    conv = Conv3x3(8)  # 28x28x1 -> 26x26x8
    pool = MaxPool2()  # 26x26x8 -> 13x13x8
    dim_maxpool = np.prod(13 * 13 * 8)
    softmax = Softmax(dim_maxpool, 10)
    
    for i in range(n_iter):

        image = test_images[i] / 255 - 0.5
        label = test_labels[i]
        feature_map, filter_conv = fun.convolute(image=image, filter_matrix=filter_conv)
        maxpool_map = fun.maxpool(feature_map=feature_map)
        probs, inter_soft = fun.softmax(maxpool_map, 
                                        weight_matrix=weight_soft, 
                                        bias_vector=bias_soft)
        weight_soft, bias_soft, gradient_soft_own = fun.backprop_softmax(inter_soft=inter_soft,
                                                           probabilities=probs,
                                                           label = label,
                                                           learn_rate=0.1) 
        gradient_max_own = fun.backprop_maxpool(feature_map=feature_map, 
                                                gradient=gradient_soft_own)
        filter_conv = fun.backprop_conv(image=image, filter_conv=filter_conv,
                                        gradient=gradient_max_own, learn_rate=0.1)
     
       
    # run model from blog with same data

        out_conv = conv.forward(image)
        #print(out_conv)   
        out_max = pool.forward(out_conv)
        out_soft, weights, summe = softmax.forward(out_max) #
        #    print(out_soft)
        gradient_L = np.zeros(10)
        gradient_L[label] = -1 / out_soft[label]
        gradient_soft, dL_dw, weights_updated, biases_updated = softmax.backprop(gradient_L, 0.1)
        gradient_max = pool.backprop(gradient_soft)
        gradient_conv, filter_update = conv.backprop(gradient_max, 0.1)
        
        
        
        ################## compare feedforward ####################################
        ###########################################################################
        print("This is iteration", i)
        if np.sum(feature_map == out_conv) == np.prod(feature_map.shape):
            print("Yeaaah! FeatureMaps are the same")
        else:
            print("featuremaps are not the same")
            
       # conv.filters == filter_conv # 
        # after first iteration these are not the same anymore,
        # since they get updated
        if np.sum(maxpool_map == out_max) == np.prod(out_max.shape):
            print("Yeaaah! maxpool is the same")
        else:
            print("maxpool is not the same")
        if np.sum(probs == out_soft) == np.prod(out_soft.shape):
            print("Yeaaah! predicted probabilities are the same")
        else:
            print("predicted probabilities are not the same")
        
        ######################### compare backprop #################################
        ############################################################################
         
        ######## softmax: gradients:    
        if np.sum(gradient_soft == gradient_soft) == np.prod(gradient_soft.shape):
            print("Yeaaah! gradients softmax are the same")
        else:
            print("gradients softmax are not the same")
        ## weight updates weight matrix softmax layer
        if np.sum(weight_soft == weights_updated) == np.prod(weight_soft.shape):
            print("Yeaaah! updated weightmatrix softmax is the same")
        else:
            print("updated weightmatrix softmax is not the same")
        ## weight updates bias vector
        if np.sum(bias_soft == biases_updated) == np.prod(biases_updated.shape):
            print("Yeaaah! Updated bias vector softmax is the same")
        else:
            print("updated bias vector is not the same")
        #### maxpool
        if np.sum(gradient_max_own== gradient_max) == np.prod(gradient_max.shape):
            print("Yeaaah! gradients maxpool layer are the same")
        else:
            print("updated gradients maxpool are not the same")
        ## conv
        if np.sum(filter_conv == filter_update) == np.prod(filter_conv.shape):
            print("Yeaaah! updated filter convlayer are the same")
        else:
            print("updated filter conv layer is not the same")
            
            
        # So! After two runs the predicted probabilities are already different, why?
        
        
    return None
    