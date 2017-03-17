#!/usr/bin/env python
# coding:utf-8

import cv2
import os
import numpy as np
from image_cropper import *  # noqa


class DataPreprocessor(object):

    # test ok
    def __init__(self, cropping_size, mean_image_path):
        self.image_cropper = ImageCropper(cropping_size)
        mean_image = np.load(mean_image_path)  # (3, 256, 256)
        self.cropped_mean_image = self.image_cropper.crop_center_image(mean_image, is_scaled=False)

    # test ok
    @staticmethod
    def image_generator(dir_path, is_verbose=False):
        for basename in os.listdir(dir_path):
            path = os.path.join(dir_path, basename)
            image = cv2.imread(path)
            if image is not None:
                name, _ = os.path.splitext(basename)
                yield image, name
            else:
                if is_verbose:
                    print("invalid image: {}".format(path))

    def save_to_one_directory(self, in_dir_path, out_dir_path):
        for image, name in DataPreprocessor.image_generator(in_dir_path, is_verbose=False):
            image = image.transpose(2, 0, 1)
            diff = self.image_cropper.crop_center_image(image, is_scaled=False) - self.cropped_mean_image
            out_path = os.path.join(out_dir_path, name + ".npy")
            np.save(out_path, diff)

    def save_images(self, input_dir_path, output_dir_path):
        for dir_path in os.listdir(input_dir_path):
            in_sub_dir_path = os.path.join(input_dir_path, dir_path)
            if not os.path.isdir(in_sub_dir_path):
                continue
            out_sub_dir_path = os.path.join(output_dir_path, dir_path)
            if not os.path.isdir(out_sub_dir_path):
                os.mkdir(out_sub_dir_path)
            self.save_to_one_directory(in_sub_dir_path, out_sub_dir_path)
