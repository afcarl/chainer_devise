#!/usr/bin/env python
# coding:utf-8

import unittest
from data_preprocessor import *  # noqa

INPUT_DIR_PATH = "../unittest_files/data_preprocessor/inputs"
OUTPUT_DIR_PATH = "../unittest_files/data_preprocessor/outputs"
MEAN_IMAGE_PATH = "../unittest_files/data_preprocessor/ilsvrc_2012_mean.npy"
CROPPING_SIZE = 227


class TestDataPreprocessor(unittest.TestCase):

    def test_save_images(self):
        preprocessor = DataPreprocessor(MEAN_IMAGE_PATH, CROPPING_SIZE)
        preprocessor.save_images(INPUT_DIR_PATH, OUTPUT_DIR_PATH)

    def test_image_generator(self):
        c = 0
        for image in DataPreprocessor.image_generator(os.path.join(INPUT_DIR_PATH, "ascospore"), is_verbose=False):
            c += 1
        self.assertTrue(c == 2)

    def test_init(self):
        preprocessor = DataPreprocessor(MEAN_IMAGE_PATH, CROPPING_SIZE)
        cropped_mean_image = preprocessor.cropped_mean_image
        self.assertTrue((3, CROPPING_SIZE, CROPPING_SIZE) == cropped_mean_image.shape)


if __name__ == "__main__":
    unittest.main()
