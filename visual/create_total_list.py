#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os.path
import random
import argparse


def child_dir_path_generator(parent_dir_path):
    for sdir in os.listdir(parent_dir_path):
        sdir_path = os.path.join(parent_dir_path, sdir)
        if not os.path.isdir(sdir_path):
            continue
        yield sdir


def create_total_list(parent_dir, child, total_list):
    list_path = os.path.join(parent_dir, child, "dataset_list.txt")
    total_list += [os.path.join(child, line.strip()) for line in open(list_path)]


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--src_dir", help="input: set a path to a source directory")
        args = parser.parse_args()
        src_dir = args.src_dir

        total_list = []
        for child in child_dir_path_generator(src_dir):
            create_total_list(src_dir, child, total_list)

        random.shuffle(total_list)
        total_list_path = os.path.join(src_dir, "total_list.txt")
        f = open(total_list_path, "w")
        for line in total_list:
            f.write(line + "\n")
    except IOError, e:
        print(e)
