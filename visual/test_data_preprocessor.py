#!/usr/bin/env python
# coding:utf-8

import unittest
from data_preprocessor import *  # noqa

INPUT_DIR_PATH = "../unittest_files/data_preprocessor/inputs"
OUTPUT_DIR_PATH = "../unittest_files/data_preprocessor/outputs"
MEAN_IMAGE_PATH = "../unittest_files/data_preprocessor/ilsvrc_2012_mean.npy"
CROPPING_SIZE = 227
NAMES = ['11_Pleospora048DFs', '12_02025k2']
INPUT_SUB_DIR_PATH = "../unittest_files/data_preprocessor/inputs/ascospore"
OUTPUT_SUB_DIR_PATH = "../unittest_files/data_preprocessor/outputs/ascospore"

# replace image by np.array


class TestDataPreprocessor(unittest.TestCase):

    def test_save_images(self):
        preprocessor = DataPreprocessor(CROPPING_SIZE, MEAN_IMAGE_PATH)
        preprocessor.save_images(INPUT_DIR_PATH, OUTPUT_DIR_PATH)

    def test_image_generator(self):
        c = 0
        names = []
        for image, name in DataPreprocessor.image_generator(os.path.join(INPUT_DIR_PATH, "ascospore"), is_verbose=False):
            c += 1
            names.append(name)
        self.assertTrue(c == 2)
        self.assertTrue(NAMES == names)

    def test_init(self):
        preprocessor = DataPreprocessor(CROPPING_SIZE, MEAN_IMAGE_PATH)
        cropped_mean_image = preprocessor.cropped_mean_image
        self.assertTrue((3, CROPPING_SIZE, CROPPING_SIZE) == cropped_mean_image.shape)

    def test_save_to_one_directory(self):
        preprocessor = DataPreprocessor(CROPPING_SIZE, MEAN_IMAGE_PATH)
        preprocessor.save_to_one_directory(INPUT_SUB_DIR_PATH, OUTPUT_SUB_DIR_PATH)


if __name__ == "__main__":
    unittest.main()
