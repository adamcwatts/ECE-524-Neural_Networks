#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
from src.activation import sigmoid, sigmoid_prime


def backprop(x, y, biases, weightsT, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.

        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and transposed weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network

        Returns:
            (nabla_b, nabla_wT): tuple containing the gradient for all the biases
                and weightsT. nabla_b and nabla_wT should be the same shape as 
                input biases and weightsT
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_wT = [np.zeros(wT.shape) for wT in weightsT]

    ### Implement here
    # feedforward
    # Here you need to store all the activations of all the units
    # by feedforward pass
    pre_act = [np.zeros(b.shape) for b in biases]  # h^k
    activations = [np.zeros(b.shape) for b in biases]  # a^k

    pre_act[0] = np.matmul(weightsT[0], x) + biases[0]  # first activation is from input layer: x = h^(k-1) -> a^k
    activations[0] = sigmoid(pre_act[0])  # hit activation layer with sigmoid function for each element h&

    for i in range(1, num_layers - 1):  # use previous activations layer output as current layers inputs
        # previous layers outputs to make new one   h^(k-1) -> a^k
        pre_act[i] = np.matmul(weightsT[i], activations[i - 1]) + biases[i]

        # hit activation layer with sigmoid function for each element: g(a^k) -> h^k
        activations[i] = sigmoid(pre_act[i])
    ###

    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer
    delta = (cost).df_wrt_a(activations[-1], y)

    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias

    L_cost_prime = (1 - y) / (1 - activations[-1]) - y / activations[-1]
    G = L_cost_prime

    for k in range(num_layers - 2, 0, -1):  # stops at 1
        G_new = np.multiply(G, sigmoid(pre_act[k]))
        nabla_b[k] = G_new
        nabla_wT[k] = np.multiply(activations[k - 1], np.transpose(G_new))
        G = np.matmul(np.transpose(weightsT[k]), G_new)

    G_new = np.multiply(G, sigmoid(pre_act[0]))
    nabla_b[0] = G_new
    nabla_wT[0] = np.multiply(activations[0], np.transpose(G_new))
    ###

    return (nabla_b, nabla_wT)
