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
    pre_act.insert(0, np.nan)
    activations = [np.zeros(b.shape) for b in biases]  # a^k
    activations.insert(0, x)

    # pre_act[0] = np.matmul(weightsT[0], x) + biases[0]  # first activation is from input layer: x = h^(k-1) -> a^k
    # activations[0] = sigmoid(pre_act[0])  # hit activation layer with sigmoid function for each element h

    for i in range(0, num_layers - 1):  # use previous activations layer output as current layers inputs
        # previous layers outputs to make new one   h^(k-1) -> a^k
        pre_act[i+1] = np.matmul(weightsT[i], activations[i]) + biases[i]
        # # hit activation layer with sigmoid function for each element: g(a^k) -> h^k
        activations[i+1] = sigmoid(pre_act[i+1])
    ###

    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer
    delta = (cost).df_wrt_a(activations[-1], y)

    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias

    # first output layer ######$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    G = delta
    nabla_b[-1] = G
    nabla_wT[-1] = np.transpose(np.matmul(activations[-2], np.transpose(G)))
    G = np.matmul(np.transpose(weightsT[-1]), G)
    # Restart Algorithm

    for n in reversed(range(0, num_layers - 2)):
        G = np.multiply(G, sigmoid_prime(pre_act[n+1]))
        nabla_b[n] = G
        nabla_wT[n] = np.transpose(np.matmul(activations[n], np.transpose(G)))
        G = np.matmul(np.transpose(weightsT[n]), G)

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    ###

    return (nabla_b, nabla_wT)
