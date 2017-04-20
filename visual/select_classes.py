#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import argparse
import os

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_file_path", help="input: a path to a file in which the number of images are decribed")
        parser.add_argument("--lower_num", type=int, help="input: lower number of images")

        args = parser.parse_args()
        num_file_path = args.num_file_path
        lower_num = args.lower_num

        classes = []
        for line in open(num_file_path):
            tokens = line.strip().split(':')
            num = int(tokens[1])
            _, tail = os.path.split(tokens[0])
            classes.append((tail, num))

        selected_classes = [x for x in classes if x[1] > lower_num]
        print(len(selected_classes))
    except IOError, e:
        print(e)
