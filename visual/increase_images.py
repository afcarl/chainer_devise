#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import argparse
import os
import sys


def make_destination_file_path(path):
    (f, ext) = os.path.splitext(path)
    return f + "_flipped" + ext


def count_images(sdir):
    return sum([1 for _ in os.listdir(sdir)])


def extract_image_paths(sdir):
    return [os.path.join(sdir, fn) for fn in os.listdir(sdir)]


def increase_images(ipaths):
    for path in ipaths:
        src_image = cv2.imread(path)
        dst_image = cv2.flip(src_image, 1)
        dst_path = make_destination_file_path(path)
        cv2.imwrite(dst_path, dst_image)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--src_dir", help="input: set a path to a source directory")
        parser.add_argument("--upper_num", help="input: the upper number of images to be increased")
        args = parser.parse_args()

        src_dir = args.src_dir
        upper_num = int(args.upper_num)
        if not os.path.exists(src_dir):
            raise IOError("{} not found".format(src_dir))

        for subdir in os.listdir(src_dir):
            subdir_path = os.path.join(src_dir, subdir)
            count = count_images(subdir_path)
            if count <= upper_num:
                print("> increased {}:{}".format(subdir, count))
                sys.stdout.flush()
                image_paths = extract_image_paths(subdir_path)
                increase_images(image_paths)
    except IOError, e:
        print(e)
