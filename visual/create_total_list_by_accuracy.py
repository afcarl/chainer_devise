#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os
import random


# test ok
def child_dir_path_generator(parent_dir_path, selected_class_path):
    for sdir in open(selected_class_path):
        sdir = sdir.strip()
        sdir_path = os.path.join(parent_dir_path, sdir)
        if not os.path.isdir(sdir_path):
            continue
        yield sdir


# test ok
def replace_labels(tmp_list, new_label):
    new_tmp_list = []
    for line in tmp_list:
        tokens = line.strip().split()
        tokens[1] = str(new_label)
        new_line = ' '.join(tokens)
        new_tmp_list.append(new_line)
    return new_tmp_list


# test ok
def make_map(path):
    label_map = {}
    for i, line in enumerate(open(path)):
        label = line.strip()
        label_map[label] = i
    return label_map


# test ok
def create_total_list(parent_dir, child, total_list, new_label):
    list_path = os.path.join(parent_dir, child, "dataset_list.txt")
    tmp_list = [os.path.join(child, line.strip()) for line in open(list_path)]
    tmp_list = replace_labels(tmp_list, new_label)
    total_list += tmp_list


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--selected_class_path", help="input: set a path to a file in which selected classes are described")
        parser.add_argument("--src_dir", help="input: set a path to a source directory")
        args = parser.parse_args()

        label_map = make_map(args.selected_class_path)
        total_list = []
        for child in child_dir_path_generator(args.src_dir, args.selected_class_path):
            new_label = label_map[child]
            create_total_list(args.src_dir, child, total_list, new_label)

        random.shuffle(total_list)
        total_list_path = os.path.join(args.src_dir, "total_list_selected.txt")
        f = open(total_list_path, "w")
        for line in total_list:
            f.write(line + "\n")

    except IOError, e:
        print(e)
