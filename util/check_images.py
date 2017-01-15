#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import os
from PIL import Image
import sys
import shutil

DIR_PATH = "/Users/kumada/Data/image_net/images"


def check_dir(dir_path):
    c = 0
    for f in os.listdir(dir_path):
        fp = os.path.join(dir_path, f)
        try:
            image = Image.open(fp)
            c += 1
        except IOError, e:
            print(" {} is not image".format(fp))
    return c


def find_invalid_files(dir_path):
    invalid_files = []
    for f in os.listdir(dir_path):
        fp = os.path.join(dir_path, f)
        try:
            image = Image.open(fp)
        except IOError, e:
            invalid_files.append(fp)
    return invalid_files


# 1. watch contents of directories.
def check_images(path):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            dir_path = os.path.join(root, d)
            count = check_dir(dir_path)
            print("{a}:{c}".format(a=dir_path, c=count))


def count_images(dir_path):
    return check_dir(dir_path)


# 2. remove directory which includes files the number of which is less than "size."
def remove_dir_less_than(size, dir_path):
    dirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path)]
    invalid_count = 0
    for d in dirs:
        count = count_images(d)
        if count < size:
            invalid_count += 1
            print(invalid_count, count, d)
            shutil.rmtree(d)


# 3,5. see directories with 100 files.
def see_valid_directories(dir_path):
    for (i, d) in enumerate(os.listdir(dir_path)):
        fd = os.path.join(dir_path, d)
        print(i, fd)
        c = check_dir(fd)
        print(c)

   
# 4. remove invalid files
def remove_invalid_files(dir_path):
    for (i, d) in enumerate(os.listdir(dir_path)):
        fd = os.path.join(dir_path, d)
        invalid_files = find_invalid_files(fd)
        for invalid_file in invalid_files:
            os.remove(invalid_file)


if __name__ == "__main__":
    see_valid_directories(DIR_PATH)
