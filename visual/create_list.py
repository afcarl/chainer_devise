#!/usr/bin/env python2.7
# -*- coding:utf-8 -*-

import os
import random
import argparse
from PIL import Image


def is_valid_image(path):
    try:
        image = Image.open(path)
        if image.mode != "RGB":
            print("{} is not RGB".format(path))
            return False
        else:
            return True
    except IOError:
        print("{} is bad image".format(path))
        return False


def extract_file_paths(path):
    names = []
    for (root, dirs, files) in os.walk(path):
        for file in files:
            path = os.path.join(root, file)
            if is_valid_image(path):
                names.append(file)
    return names


def create_list(parent_dir, dir_name, label):
    src_path = os.path.join(parent_dir, dir_name)
    dst_path = os.path.join(parent_dir, dir_name, "list.txt")
    names = extract_file_paths(src_path)
    random.shuffle(names)
    with open(dst_path, "w") as f:
        for name in names:
            f.write("{} {}\n".format(name, label))


def make_label_map(dir_path, label_path):
    label_map = {}
    with open(label_path, "w") as outf:
        label = 0
        for sdir in os.listdir(dir_path):
            sdir_path = os.path.join(dir_path, sdir)
            if os.path.isdir(sdir_path):
                outf.write("{} {}\n".format(sdir, label))
                label_map[sdir] = label
                label += 1
    return label_map


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--src_dir", help="input: set a path to a source directory")
        parser.add_argument("--label_path", help="output: set a path to a label file")

        args = parser.parse_args()
        src_dir = args.src_dir
        label_path = args.label_path
        label_map = make_label_map(src_dir, label_path)
        for (key, value) in label_map.items():
            create_list(src_dir, key, value)
    except IOError, e:
        print("EXCEPT")
        print(e)
