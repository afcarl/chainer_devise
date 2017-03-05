#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import argparse
import os
import cv2
import sys


def path_generator(dir_path, is_path):
    for item in os.listdir(dir_path):
        path = os.path.join(dir_path, item)
        if is_path(path):
            yield path


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--src_dir", help="input: set a path to a source directory")
        parser.add_argument("--dst_dir", help="output: set a path to a destination directory")
        parser.add_argument("--size", help="input: size")

        args = parser.parse_args()
        src_dir = args.src_dir
        dst_dir = args.dst_dir
        size = int(args.size)

        print("src_dir: {}".format(src_dir))
        print("dst_dir: {}".format(dst_dir))
        print("size: {}".format(size))

        if not os.path.exists(src_dir):
            raise IOError("{} not found".format(src_dir))

        if os.path.exists(dst_dir):
            raise IOError("{} already exists".format(dst_dir))

        os.mkdir(dst_dir)
        sys.stdout.flush()
        for image_dir in path_generator(src_dir, os.path.isdir):
            print("> src image dir {}".format(image_dir))
            sys.stdout.flush()
            dst_image_dir = os.path.join(dst_dir, os.path.basename(image_dir))
            print("> dst image dir {}".format(dst_image_dir))
            sys.stdout.flush()
            os.mkdir(dst_image_dir)
            for src_file_path in path_generator(image_dir, os.path.isfile):
                print(" >> src image path: {}".format(src_file_path))
                sys.stdout.flush()
                src_image = cv2.imread(src_file_path)
                if src_image is None:
                    print("invalid image")
                    continue
                dst_image = cv2.resize(src_image, (size, size))
                basename = os.path.basename(src_file_path)
                _, ext = os.path.splitext(basename)
                lower_ext = ext.lower()
                print(">>>", lower_ext)
                if lower_ext == ".jpg" or lower_ext == ".jpeg" or lower_ext == ".tif" or lower_ext == ".png":
                    pass
                else:
                    basename += ".jpg"

                dst_image_path = os.path.join(dst_image_dir, basename)
                print(" >> dst image path: {} : {}".format(dst_image_path, ext))
                sys.stdout.flush()
                cv2.imwrite(dst_image_path, dst_image)
    except IOError, e:
        print(e)
