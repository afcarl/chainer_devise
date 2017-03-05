#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import os
from PIL import Image
import shutil


def check_dir(dir_path, is_verbose=True):
    c = 0
    for f in os.listdir(dir_path):
        fp = os.path.join(dir_path, f)
        try:
            Image.open(fp)
            c += 1
        except IOError:
            if is_verbose:
                print(" {} is not image".format(os.path.basename(fp)))
            else:
                pass
    return c


def find_invalid_files(dir_path):
    invalid_files = []
    for f in os.listdir(dir_path):
        fp = os.path.join(dir_path, f)
        try:
            Image.open(fp)
        except IOError:
            invalid_files.append(fp)
    return invalid_files


def convert_to_rgb_images_in_each_directory(dir_path):
    invalid_files = []
    for f in os.listdir(dir_path):
        fp = os.path.join(dir_path, f)
        try:
            image = Image.open(fp)  # exception may occur
            if image.mode != "RGB":
                image = image.convert("RGB")  # exception may occur
                a, _ = os.path.splitext(fp)
                new_fp = a + ".jpg"
                print("file {} is not RGB -> {}".format(fp, new_fp))
                image.save(new_fp)
                os.remove(fp)
            else:
                pass  # print("file {} is RGB".format(fp))
        except IOError:
            invalid_files.append(fp)
    return invalid_files


def convert_to_rgb(dir_path):
    for d in os.listdir(dir_path):
        fd = os.path.join(dir_path, d)
        a, b = os.path.splitext(fd)
        if not os.path.isdir(fd):
            continue
        invalid_files = convert_to_rgb_images_in_each_directory(fd)
        if len(invalid_files) != 0:
            for f in invalid_files:
                print("This image either cannot be loaded or cannot be converted to RGB:{} -> remove it".format(f))
                os.remove(f)


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


# 3,5. see directories with the number of files.
def see_directories(dir_path):
    for (i, d) in enumerate(os.listdir(dir_path)):
        fd = os.path.join(dir_path, d)
        c = check_dir(fd, is_verbose=False)
        print("{i} {d} {c}".format(i=i, d=d, c=c))


# 4. remove invalid files
def remove_invalid_files(dir_path):
    for (i, d) in enumerate(os.listdir(dir_path)):
        fd = os.path.join(dir_path, d)
        if not os.path.isdir(fd):
            continue
        invalid_files = find_invalid_files(fd)
        for invalid_file in invalid_files:
            print("remove {}".format(invalid_file))
            os.remove(invalid_file)


# select directories each of which includes files >= lower_size.
def select_directories(dir_path, lower_size):
    list_path = os.path.join(dir_path, "list")
    for line in open(list_path):
        items = line.strip().split()
        size = int(items[2])
        if lower_size <= size:
            print(items[1])


def make_histogram(dir_path, lower_size):
    list_path = os.path.join(dir_path, "list")
    sizes = []
    for line in open(list_path):
        items = line.strip().split()
        size = int(items[2])
        if lower_size <= size:
            sizes.append(size)
    max_size = 2000
    step_size = 10
    bin_size = max_size / step_size
    histogram = [0] * bin_size
    for size in sizes:
        index = size / step_size
        histogram[index] += 1

    print(sum(histogram))
    for v in histogram:
        print(v)


if __name__ == "__main__":
    check_images("/Volumes/TOSHIBA EXT/mac/image_net/selected_images_256")
