#!/usr/bin/env python
# coding:utf-8

import unittest
from data_preprocessor_for_devise import DataPreprocessorForDevise
import os
import numpy as np


class TestDataPreprocessorForDevise(unittest.TestCase):

    def construct_instance(self):
        root_dir_path = '/home/ubuntu/data/devise/selected_images_256_greater_than_200_images'
        training_path = os.path.join(root_dir_path, 'train_valid_selected_.txt')
        mean_image_path = '/home/ubuntu/libs/caffe-master/python/caffe/imagenet/ilsvrc_2012_mean.npy'
        crop_size = 227
        mean = np.load(mean_image_path)
        preprocessor = DataPreprocessorForDevise(
            training_path,
            root_dir_path,
            mean,
            crop_size,
            random=False,
            is_scaled=True
        )
        return preprocessor

    def test_init(self):
        preprocessor = self.construct_instance()
        self.assertTrue(preprocessor is not None)

    def test_len(self):
        preprocessor = self.construct_instance()
        self.assertTrue(len(preprocessor) == 49628)


if __name__ == '__main__':
    unittest.main()
