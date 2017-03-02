#!/usr/bin/env python
# coding:utf-8


import cv2
import numpy as np
import os


class ImageCropper(object):
    INITIAL_IMAGE_SIZE = 256
    SCALE_FACTOR = 255

    def __init__(self, in_size):
        self.in_size = in_size
        self.cropwidth = self.INITIAL_IMAGE_SIZE - self.in_size

    def crop(self, img, top, left):
        bottom = self.in_size + top
        right  = self.in_size + left
        return img[:, top:bottom, left:right].astype(np.float32)

    # test ok
    def crop_center_image(self, image, is_scaled=True):
        top  = self.cropwidth / 2
        left = self.cropwidth / 2
        image = self.crop(image, top, left)
        if is_scaled:
            image /= self.SCALE_FACTOR
        return image

    # test ok
    def crop_center(self, path, is_scaled=True):
        # Data loading routine
        image = cv2.imread(path).transpose(2, 0, 1)
        return self.crop_center_image(image, is_scaled)


import unittest
import subprocess

class ImageCropperTest(unittest.TestCase):

    def test_crop_center(self):
        SAMPLE_IMAGE_PATH = "/home/ubuntu/data/mytest/input/sample.png"
        insize = 227
        cropper = ImageCropper(insize)
        center_image = cropper.crop_center(SAMPLE_IMAGE_PATH, is_scaled=False)
        SAVING_PATH = "/home/ubuntu/data/mytest/output/center.png"
        answer = cv2.imread(SAVING_PATH).transpose(2, 0, 1).astype(np.float32)
        self.assertTrue(np.all(answer == center_image))        

    def test_crop_center_image(self):
        SAMPLE_IMAGE_PATH = "/home/ubuntu/data/mytest/input/sample.png"
        image = cv2.imread(SAMPLE_IMAGE_PATH).transpose(2, 0, 1)
        insize = 227
        cropper = ImageCropper(insize)
        image = cropper.crop_center_image(image, is_scaled=False)
        self.assertTrue((3, insize, insize) == image.shape) 

        SAVING_PATH = "/home/ubuntu/data/mytest/output/center.png"
        answer = cv2.imread(SAVING_PATH).transpose(2, 0, 1).astype(np.float32)
        self.assertTrue(np.all(answer == image))        

if __name__ == "__main__":
    unittest.main()   
   
