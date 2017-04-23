#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import argparse
import os
import shutil


def check_path(path):
    if not os.path.exists(path):
        raise Exception('invalid path')


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_file_path", help="input: a path to a file in which the number of images are decribed")
        parser.add_argument("--lower_num", type=int, help="input: lower number of images")
        parser.add_argument("--input_dir_path",  help="input: a path to an input directory path")
        parser.add_argument("--output_dir_path", help="output: a path to an output directory path")

        args = parser.parse_args()
        num_file_path = args.num_file_path
        check_path(num_file_path)
        lower_num = args.lower_num
        input_dir_path = args.input_dir_path
        output_dir_path = args.output_dir_path
        check_path(input_dir_path)
        check_path(output_dir_path)

        classes = []
        for line in open(num_file_path):
            tokens = line.strip().split(':')
            num = int(tokens[1])
            _, tail = os.path.split(tokens[0])
            classes.append((tail, num))

        selected_classes = [x for x in classes if x[1] >= lower_num]
        print('the number of classes: {}'.format(len(selected_classes)))
        for selected_class in selected_classes:
            input_sub_dir_path = os.path.join(input_dir_path, selected_class[0])
            output_sub_dir_path = os.path.join(output_dir_path, selected_class[0])
            # print('{} -> {}'.format(input_sub_dir_path, output_sub_dir_path))
            shutil.copytree(input_sub_dir_path, output_sub_dir_path)

    except IOError, e:
        print(e)
