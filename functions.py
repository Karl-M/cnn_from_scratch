#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 19:40:43 2019

@author: konstantin
"""
# collection of functions

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

def copy_images(source, dest, first, last):
    """ source: directory containing original data
        dest: target directory
        first: number of first image to copy (e.g. '0' for 0.jpg)
        last: number of last image to copy (e.g. '1000' for 1000.jpg)
    """
    fnames = [str(i) + ".jpg" for i in range(first, last)]
    for name in fnames:
        src = os.path.join(source, name)
        dst = os.path.join(dest, name)
        shutil.copyfile(src, dst)
    return


def extract_features(directory, datagen, conv_base, sample_count, batch_size):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))

    generator = datagen.flow_from_directory(directory,
                                        target_size=(150, 150),
                                        class_mode="binary",
                                        batch_size=batch_size)

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels



def plot_accuracy(history):
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc,  "bo", label="training acc")
    plt.plot(epochs, val_acc,  "b", label="validation acc")
    plt.title("training and validation accuracy")
    plt.show()
    return


def plot_loss(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss,  "bo", label="training loss")
    plt.plot(epochs, val_loss,  "b", label="validation loss")
    plt.title("training and validation loss")
    plt.show()
    return

#### nn from scratch functions
    

def feed_forward(model, data):
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
    z1 = data.dot(W1) + b1
    a1 : np.ndarray = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2) # for softmax
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) #probabilities
    model = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    intermediates = {"z1": z1, "a1": a1, "z2": z2} 
    return probs, model, intermediates



def backprop(probs, model, intermediates, data):
    W1, b1, W2, b2 = model["W1"], model["b1"], model["W2"], model["b2"]
    z1 = intermediates["z1"]
    a1 = intermediates["a1"]
    z2 = intermediates["z2"]
   # Backpropagation
    delta3 = probs
    delta3[range(num_examples), y] -= 1
    dW2 = (a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0)
    delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
    dW1 = np.dot(data.T, delta2)
    db1 = np.sum(delta2, axis=0)
    
    # update weights
    W1 += -nu * dW1
    b1 += -nu * db1
    W2 += -nu * dW2
    b2 += -nu * db2    

    model["W1"], model["b1"], model["W2"], model["b2"] = W1, b1, W2, b2
    return model





def training(n_iter, data, print_acc=True, print_loss=True, mini_batch_size=1):
    W1 = np.random.randn(nn_input_dim, h_dim)
    b1 = np.random.randn(h_dim)
    W2 = np.random.randn(h_dim, nn_output_dim)
    b2 = np.random.randn(nn_output_dim)
    
    model = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    #data = X    
    for iter in range(n_iter):
        probs, model, intermediates = feed_forward(model, X)
        ## stochastic gradient implementation
        if mini_batch_size == 1:
            model = backprop(probs, model, intermediates, data)
        else:
            batch_size = round(len(X) * mini_batch_size)
            batch = np.random.choice(range(len(X)), replace=False, size=batch_size)
            model = backprop(probs, model, intermediates, batch)    
        if print_acc and iter % 1000 == 0:
            predictions = np.argmax(feed_forward(model, X)[0], axis=1)
            print(f"accuracy bei iteration {iter}", np.mean(predictions == y))
        if print_loss and iter % 1000 == 0:
            print(f"loss bei iteration {iter}", calculate_loss(model))
            
    return model




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


def max_pool(feature_map):
    number_filters, width, height = feature_map.shape
#    print(feature_map.shape)
    pooling_map = np.zeros(shape=(number_filters, width // 2, height // 2))
    print(pooling_map.shape)
    for k in range(number_filters):
       for i in range(int(width / 2)):
            for j in range(int(height / 2)):
                res = feature_map[k, i*2:i*2 + 2, j*2:(j*2 + 2)]
                pooling_map[k, i, j] = np.max(res)
                
    return pooling_map

