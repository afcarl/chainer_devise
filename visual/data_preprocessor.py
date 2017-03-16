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
        self.cropped_mean_image = self.image_cropper.crop_center_image(mean_image, is_scale=False)

    # test ok
    @staticmethod
    def image_generator(dir_path, is_verbose=False):
        for item in os.listdir(dir_path):
            path = os.path.join(dir_path, item)
            image = cv2.imread(path)
            if image is not None:
                yield image
            else:
                if is_verbose:
                    print("invalid image: {}".format(path))

    def save_to_one_directory(self, dir_path):
        for image in DataPreprocessor.image_generator(dir_path, is_verbose=False):
            pass

    def save_images(self, input_dir_path, output_dir_path):
        pass
