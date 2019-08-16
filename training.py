# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 13:12:12 2019

@author: d2gu53
"""

# convolution helps us look for specific localized image features 
import numpy as np
import sys
#from pathlib import Path
import mnist
import os
#path = "/home/konstantin/Documents/master_arbeit/cnn_from_scratch"
#path2 = "/home/konstantin/Documents/master_arbeit/cnn-from-scratch"
path = "C:/Users/D2GU53/Documents/master_arbeit/nn_in_r"
sys.path.append(path)
#sys.path.append(path2)

import functions as fun
import training as train


def training(n_iter, n_classes, n_filter, training_data, label, 
             learn_rate=0.01, print_acc=True, weights_conv=None, weights_soft=None):
    
#    permutation = np.random.permutation(len(train_images))
#    train_images = train_images[permutation]
#    train_labels = train_labels[permutation]

    num_correct = 0
    input_dim = int((((training_data[0].shape[0] - 3 + 1) / 2) ** 2) * n_filter)
    input_dim = 13 * 13 * 8
    if weights_conv == None:
        np.random.seed(seed=666); filter_matrix_conv = np.random.randn(n_filter, 3, 3) * 3
       # bias_vector_conv = np.random.randn(n_filter) / n_filter
    else:
        filter_matrix_conv = weights_conv["weight_matrix"]
        
    if weights_soft == None:
        np.random.seed(seed=666); weight_matrix_soft = (np.random.randn(input_dim, n_classes) / input_dim) * 10
        np.random.seed(seed=666); bias_vector_soft = np.random.randn(n_classes)
    else:
        weight_matrix_soft = weights_soft["weight_matrix"]
        bias_vector_soft= weights_soft["bias_vector"]
        
    for i in range(n_iter):
        
        if i % 10 == 0:
            print("this is iteration: ", i)
        image = training_data[i] / 255 - 0.5
        
        out_conv, filter_mat, intermediates_conv = fun.convolute(
                image=image, 
                filter_matrix=filter_matrix_conv
                )
     #   print(out_conv)
        
        out_maxpool = fun.max_pool(feature_map=out_conv)
      #  print("maxpool filter 0: ", out_maxpool[0])
        probabilities, intermediates_soft = fun.softmax(
                output_maxpool=out_maxpool, 
                weight_matrix=weight_matrix_soft,
                bias_vector=bias_vector_soft)
        
        print(probabilities)
        
        weight_matrix_soft, bias_vector_soft, deltaL_cor = fun.backprop_softmax(
                probabilities=probabilities,
                inter_soft=intermediates_soft,
                label=label[i],
                learn_rate=learn_rate)
        
        feature_map_back = fun.backprop_maxpool(feature_map=out_conv, 
                                            gradient=deltaL_cor)
#        
        filter_matrix_conv, grads = fun.backprop_conv(image, filter_mat, feature_map_back, learn_rate)
       # print(filter_matrix_conv.shape)
        #print(grads.shape)

        prediction = np.argmax(probabilities)
        acc = 1 if prediction == label[i] else 0
        num_correct += acc
        
        if i % 100 == 0 and i != 0 and print_acc:
            accuracy = num_correct / i
            print(f"accuracy for the first {i} samples: {accuracy}")
            print(f"{num_correct} predictions for {i} samples were correct")
            print(filter_matrix_conv[7])
            
    weights_conv = {"weight_matrix": filter_matrix_conv}
    
    weights_soft = {"weight_matrix": weight_matrix_soft,
                    "bias_vector": bias_vector_soft}

    return weights_conv, weights_soft, feature_map_back, out_conv
