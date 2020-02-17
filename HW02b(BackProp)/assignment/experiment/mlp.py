#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act

DATA_PATH = '../../data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')


    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784, 20, 15, 10])  # modified
    # model = network2.Network([784, 20, 10])  # original
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=3, weight_id=3)   # modified
    # model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)  # original


def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    model = network2.Network([784, 100,  10])
    # train the network using SGD
    epochs = 100
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = model.SGD(
        training_data=train_data,
        epochs=epochs,
        mini_batch_size=128,
        eta=1e-3,
        lmbda = 0,
        evaluation_data=test_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
    plotter(evaluation_cost, evaluation_accuracy, training_cost, training_accuracy)


def plotter(val_cost, val_accuracy, train_cost, train_accuracy):
    SMALL_SIZE = 20
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 26
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    epochs = len(val_cost)

    # Training plots
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    axe.scatter(np.arange(0, epochs), train_cost)
    axe.set_ylabel('Training Cost', labelpad=15)
    axe.set_xlabel('Epochs', labelpad=15)
    plt.tight_layout()
    plt.savefig('training_cost.png')
    plt.show()
    # plt.close()

    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    axe.scatter(np.arange(0, epochs), train_accuracy)
    axe.set_ylabel('Training Accuracy', labelpad=15)
    axe.set_xlabel('Epochs', labelpad=15)
    axe.set_ylim((0, 3_000))
    plt.tight_layout()
    plt.savefig('training_acc.png')
    plt.show()
    # plt.close()

    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    axe.scatter(np.arange(0, epochs), val_cost)
    axe.set_ylabel('Validation Cost', labelpad=15)
    axe.set_xlabel('Epochs', labelpad=15)
    plt.tight_layout()
    plt.savefig('valid_cost.png')
    plt.show()
    # plt.close()

    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    axe.scatter(np.arange(0, epochs), val_accuracy)
    axe.set_ylabel('Validation Accuracy', labelpad=15)
    axe.set_xlabel('Epochs', labelpad=15)
    axe.set_ylim((0, 10_000))
    plt.tight_layout()
    plt.savefig('valid_acc.png')
    plt.show()
    # plt.close()

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()
