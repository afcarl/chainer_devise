#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-


import argparse


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()

        parser.add_argument('--gpu', default=0, type=int, help='INPUT: GPU ID (negative value indicates CPU)')
        parser.add_argument('--root_dir_path',
                            help='INPUT: a path to a training/testing directory')
        parser.add_argument('--training_path',
                            help='INPUT: a path to a file in which all paths needed for training are described')
        parser.add_argument('--testing_path',
                            help='INPUT: a path to a file in which all paths needed for testing are described')
        parser.add_argument('--output_dir_path', help='OUTPUT: a path to an output directory')
        parser.add_argument('--batch_size', default=32, type=int, help='INPUT: minibatch size')
        parser.add_argument('--epoch_size', default=10, type=int, help='INPUT: number of epochs to train')
        parser.add_argument('--log_interval', type=int, help='input: test interval')
        parser.add_argument('--model_epoch', type=int, default=1, help='input: epoch to save model')
        args = parser.parse_args()

    except Exception, e:
        print(e)
