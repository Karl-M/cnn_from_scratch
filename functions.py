#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:40:43 2019

@author: konstantin
"""
# collection of functions

#import os
#import shutil
import numpy as np
#import matplotlib.pyplot as plt



# convolution operation with randomnly initialized filters
def convolute(image, filter_matrix, bias_vector):
    
    number_filters = filter_matrix.shape[0]
    height, width = image.shape
    feature_map = np.zeros(shape=(number_filters, height - 3 + 1, width - 3 + 1))

    for k in range(number_filters):
       for i in range(height - 3 + 1):
            for j in range(width - 3 + 1):
                res = image[i:(i + 3), j:(j + 3)] * filter_matrix[k]
                feature_map[k, i, j] = np.sum(res) + bias_vector[k]
    
    intermediates = {"num_fil": number_filters, "height": height,
                          "width": width}
    
    return feature_map, intermediates


def max_pool(feature_map):
    if len(feature_map.shape) == 3:
        number_filters, height, width = feature_map.shape
    else:
        number_filters = 0
        height, width = feature_map.shape
#    print(feature_map.shape)
    pooling_map = np.zeros(shape=(number_filters, height // 2, width // 2))
    for k in range(number_filters):
       for i in range(height // 2):
            for j in range(width // 2):
                res = feature_map[k, i*2:i*2 + 2, j*2:(j*2 + 2)]
                pooling_map[k, i, j] = np.max(res)
                
    return pooling_map



# softmax

def softmax(output_maxpool, weight_matrix, bias_vector):
    
    n_classes = weight_matrix.shape[1]
    num_filter, height, width = output_maxpool.shape
    input_len = num_filter * height * width
    output_maxpool_flattened = np.reshape(output_maxpool, 
                                newshape=(input_len))
    
  #  bias_vector = np.zeros(shape=10)
  #  weight_matrix = np.ones(shape=(1014, 10))
    input_softmax = output_maxpool_flattened.dot(weight_matrix) + bias_vector
    
    exponentials = np.exp(input_softmax)
    sum_exponentials = np.sum(exponentials)
    probabilities = exponentials / sum_exponentials
    
    intermediates = {"exp": exponentials, 
                             "sum_exp": sum_exponentials,
                             "input_softmax": input_softmax,
                             "weight_matrix": weight_matrix,
                             "output_maxpool_flattened": output_maxpool_flattened,
                             "bias_vector": bias_vector,
                             "n_classes": n_classes
                             }
    return probabilities, intermediates


def backprop_softmax(inter_soft, probabilities, label, learn_rate=0.01):
    ps = probabilities
    
    pooling_map_shape = inter_soft["output_maxpool_flattened"].shape
    # to implement backprop, we need intermediate results, 
    # e.g. the derivative of the loss
    
    # derivative of loss function with respect to output last layer
    dLoss_daL = np.zeros(inter_soft["n_classes"]) # dL / da
    dLoss_daL[label] = - 1 / ps[label]

    # derivative of softmax with respect to input 
    # (input =  - (output_maxpool.dot(weight_matrix) + bias_vector) )
    daL_dzL = np.zeros(inter_soft["n_classes"])
#    daL_dzL = ((inter_soft["exp"][label] * inter_soft["exp"]) / 
#               (inter_soft["sum_exp"] ** 2))
    daL_dzL[label] = ((inter_soft["exp"][label] * 
           (- inter_soft["exp"][label] + inter_soft["sum_exp"])) /
        ( inter_soft["sum_exp"] ** 2) )

    # derivative of Loss with respect to bias vector in softmax
    deltaL = np.zeros(inter_soft["n_classes"])
    deltaL[label] = dLoss_daL[label] * daL_dzL[label]
    #dbL[label] = dLoss[label]
    dL_dbL = deltaL
    
    # derivative with respect to weight matrix in softmax
    dzL_dwL = np.zeros(shape=(np.prod(pooling_map_shape), inter_soft["n_classes"]))
    dzL_dwL[:, label] = inter_soft["output_maxpool_flattened"]
    
    # derivative of Loss function with respect to weight matrix in softmax
    dL_dwL = np.zeros(shape=(np.prod(pooling_map_shape), inter_soft["n_classes"]))
    #dL_dwL[:, label] = dzL_dwL.dot(deltaL) # version from blog    
    dL_dwL[:, label] = np.dot(deltaL[label], inter_soft["output_maxpool_flattened"]) # my version
    # updating weights
    weight_matrix = inter_soft["weight_matrix"] - learn_rate * dL_dwL 
    bias_vector = inter_soft["bias_vector"] - learn_rate * dL_dbL
    

    return weight_matrix, bias_vector


def training(n_iter, n_classes, n_filter, training_data, label, 
             learn_rate=0.01, print_acc=True, weights_conv=None, weights_soft=None):
    
    num_correct = 0
    input_dim = int((((training_data[0].shape[0] - 3 + 1) / 2) ** 2) * n_filter)
    
    if weights_conv == None:
        filter_matrix_conv = np.random.randn(n_filter, 3, 3) / 9
        bias_vector_conv = np.random.randn(n_filter) / n_filter
    else:
        filter_matrix_conv = weights_conv["weight_matrix"]
        bias_vector_conv = weights_conv["bias_vector"]
        
    if weights_soft == None:
        weight_matrix_soft = np.random.randn(input_dim, n_classes) / (input_dim)
        bias_vector_soft = np.random.randn(n_classes) / (n_classes)
    else:
        weight_matrix_soft = weights_soft["weight_matrix"]
        bias_vector_soft= weights_soft["bias_vector"]
    
    for i in range(n_iter):
        image = training_data[i] / 255 - 0.5
        
        out_conv, intermediates_conv = convolute(
                image=image, 
                filter_matrix=filter_matrix_conv,
                bias_vector=bias_vector_conv)
        
        out_maxpool = max_pool(feature_map=out_conv)
        
        probabilities, intermediates_soft = softmax(
                output_maxpool=out_maxpool, 
                weight_matrix=weight_matrix_soft,
                bias_vector=bias_vector_soft)
        
        weight_matrix_soft, bias_vector_soft = backprop_softmax(
                probabilities=probabilities,
                inter_soft=intermediates_soft,
                label=label[i],
                learn_rate=learn_rate)
        
        #print(out_maxpool.shape)
  #      loss = -np.log(probabilities[label])
        prediction = np.argmax(probabilities)
     #   print(f"prediction: {prediction}")
     #   print(f"label: {label[i]}")
        acc = 1 if prediction == label[i] else 0
        num_correct += acc
        
        if i % 100 == 0 and i != 0 and print_acc:
            accuracy = num_correct / i
            print(f"accuracy for the first {i} samples: {accuracy}")
            print(f"{num_correct} predictions for {i} samples were correct")
      #      print(bias_vector_conv)
       #     print(bias_vector_soft)
            
    weights_conv = {"weight_matrix": filter_matrix_conv, 
                    "bias_vector": bias_vector_conv}
    
    weights_soft = {"weight_matrix": weight_matrix_soft,
                    "bias_vector": bias_vector_soft}

    return weights_conv, weights_soft
    


