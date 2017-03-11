#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import argparse
from create_list import *  # noqa


def count_images(dir_path):
    c = 0
    for f in os.listdir(dir_path):
        fpath = os.path.join(dir_path, f)
        if is_valid_image(fpath):
            c += 1
    return c


def create_dataset(path, training_rate, testing_rate):
    total_count = count_images(path)
    train_size = int(total_count * training_rate)
    test_size = int(total_count * testing_rate)
    valid_size = total_count - train_size - test_size

    dataset_list_path = os.path.join(path, "dataset_list.txt")
    with open(dataset_list_path, "w") as dataset_list:
        list_path = os.path.join(path, "list.txt")
        for (i, line) in enumerate(open(list_path)):
            line = line.strip()
            if i < train_size:
                dataset_list.write("{p} train\n".format(p=line))
            elif train_size <= i and i < train_size + valid_size:
                dataset_list.write("{p} valid\n".format(p=line))
            elif train_size + valid_size <= i and i < train_size + valid_size + test_size:
                dataset_list.write("{p} test\n".format(p=line))
    return (train_size, valid_size, test_size)


def check_dataset(path):
    list_path = os.path.join(path, "dataset_list.txt")
    train_count = 0
    valid_count = 0
    test_count = 0
    for line in open(list_path):
        if " train" in line:
            train_count += 1
        elif " valid" in line:
            valid_count += 1
        else:
            test_count += 1

    return train_count, valid_count, test_count


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--src_dir", help="input: set a path to a source directory")
        parser.add_argument("--training", type=int, help="input: integer value for training")
        parser.add_argument("--testing", type=int, help="input: integer value for testing")
        parser.add_argument("--validation", type=int, help="input: integer value for validation")

        args = parser.parse_args()
        src_dir = args.src_dir
        training = args.training
        testing = args.testing
        validation = args.validation
        total = float(training + testing + validation)
        training_rate = training / total
        testing_rate = testing / total
        validation_rate = validation / total

        for sdir in os.listdir(src_dir):
            dir_path = os.path.join(src_dir, sdir)
            if not os.path.isdir(dir_path):
                continue
            train_a, valid_a, test_a = create_dataset(dir_path, training_rate, testing_rate)
            train_b, valid_b, test_b = check_dataset(dir_path)
            assert train_a == train_b, "dir:{} {}-{}".format(dir_path, train_a, train_b)
            assert valid_a == valid_b, "dir:{} {}-{}".format(dir_path, valid_a, valid_b)
            assert test_a == test_b, "dir:{} {}-{}".format(dir_path, test_a, test_b)

    except IOError, e:
        print(e)
