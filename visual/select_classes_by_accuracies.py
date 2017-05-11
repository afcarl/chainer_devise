#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import argparse
import re
import os

PATTERN = re.compile(r'^(\d+):')


def make_map(path):
    results = {}
    for line in open(path):
        tokens = line.strip().split()
        results[int(tokens[1])] = tokens[0]
    return results


def accuracy_generator(path, lower_accuracy):
    for line in open(path):
        tokens = line.strip().split()
        if len(tokens) == 2:
            m = PATTERN.findall(tokens[0])
            if len(m) == 1:
                label = int(m[0])
                accuracy = float(tokens[1])
                if accuracy >= lower_accuracy:
                    yield label, accuracy


def calculate_average_accuracy(path, lower_accuracy):
    total_accuracy = 0.0
    total_count = 0
    for (label, accuracy) in accuracy_generator(path, lower_accuracy):
        total_accuracy += accuracy
        total_count += 1

    average_accuracy = total_accuracy / total_count
    print(lower_accuracy, average_accuracy, total_count)


def select_classes(accuracy_path, lower_accuracy, label_path, output_path):
    label_map = make_map(label_path)
    with open(output_path, 'w') as fout:
        for (label, accuracy) in accuracy_generator(accuracy_path, lower_accuracy):
            fout.write('{}\n'.format(label_map[label]))


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--accuracy_path", help="input: a path to a file in which accuracies are described")
        parser.add_argument("--lower_accuracy", type=float, help="input: lower accuracy")
        parser.add_argument("--input_dir_path",  help="input: a path to an input directory path")
        parser.add_argument("--output_path", help="output: a path to a output file")
        parser.add_argument("--mode", help="input: 'check' or 'select'")

        args = parser.parse_args()
        if args.mode == 'check':
            calculate_average_accuracy(args.accuracy_path, args.lower_accuracy)
        elif args.mode == 'select':
            label_path = os.path.join(args.input_dir_path, 'label.txt')
            select_classes(args.accuracy_path, args.lower_accuracy, label_path, args.output_path)
        else:
            raise IOError('invalid mode')
    except IOError, e:
        print(e)
