#!/usr/bin/env python
# coding:utf-8

from data_preprocessor import *  # noqa
import argparse
CROPPING_SIZE = 227

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--mean_image_path", help="input: set a path to a mean image")
        parser.add_argument("--input_dir_path", help="input: set a path to an input dir path")
        parser.add_argument("--output_dir_path", help="output: set a path to an input dir path")

        args = parser.parse_args()
        mean_image_path = args.mean_image_path
        input_dir_path = args.input_dir_path
        output_dir_path = args.output_dir_path

        preprocessor = DataPreprocessor(CROPPING_SIZE, mean_image_path)
        if not os.path.isdir(output_dir_path):
            os.mkdir(output_dir_path)

        preprocessor.save_images(input_dir_path, output_dir_path)

    except IOError, e:
        print(e)
