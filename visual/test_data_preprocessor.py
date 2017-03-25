#!/usr/bin/env python
# coding:utf-8


import unittest
import data_preprocessor
from chainer import datasets
import numpy as np
import os
import cv2
import shutil

TEST_INPUT_DIR_PATH = "../unittest_input"
TEST_OUTPUT_DIR_PATH = "../unittest_output"
MEAN_NAME = "meang.npy"
MEAN_VALUE = 3
FILE_NAMES = ["1", "2"]
SUB_DIR_NAMES = ["hoge", "foo"]
TEXT_FILE_NAME = "file.txt"
IN_SIZE = 227
EPSILON = 1.0e-08


class TestDataPreprocessor(unittest.TestCase):

    # make input dirs and input files for unit test
    def setUp(self):
        sub_dir_paths = TestDataPreprocessor.make_dirs()
        for sub_dir_path in sub_dir_paths:
            for file_name in FILE_NAMES:
                image = TestDataPreprocessor.make_test_image(int(file_name))
                image_path = os.path.join(sub_dir_path, file_name + ".png")
                cv2.imwrite(image_path, image)
        mean_image = TestDataPreprocessor.make_test_image(MEAN_VALUE).transpose(2, 0, 1)
        mean_image_path = os.path.join(TEST_INPUT_DIR_PATH, MEAN_NAME)
        np.save(mean_image_path, mean_image)
        TestDataPreprocessor.make_text_file()

    @staticmethod
    def make_text_file():
        path = os.path.join(TEST_INPUT_DIR_PATH, TEXT_FILE_NAME)
        with open(path, "w") as fin:
            k = 0
            for subdir in SUB_DIR_NAMES:
                for name in FILE_NAMES:
                    line = "{}/{}.png {}\n".format(subdir, name, k)
                    fin.write(line)
                    k += 1

    @staticmethod
    def make_test_image(value):
        image = np.zeros((256, 256, 3))
        image[14:241, 14:241, :] = value
        return image

    @staticmethod
    def make_dirs():
        sub_dir_paths = [os.path.join(TEST_INPUT_DIR_PATH, name) for name in SUB_DIR_NAMES]
        for sub_dir_path in sub_dir_paths:
            if not os.path.isdir(sub_dir_path):
                os.makedirs(sub_dir_path)
        return sub_dir_paths

    def test_LabeledImageDataset(self):
        file_path = os.path.join(TEST_INPUT_DIR_PATH, TEXT_FILE_NAME)
        root = TEST_INPUT_DIR_PATH
        dataset = datasets.LabeledImageDataset(file_path, root)
        self.assertTrue(len(dataset) == 4)
        answer_labels = [0, 1, 2, 3]
        for answer_label, (image, label) in zip(answer_labels, dataset):
            self.assertTrue(answer_label == label)
            self.assertTrue(image.shape == (3, 256, 256))

    def test_init(self):
        file_path = os.path.join(TEST_INPUT_DIR_PATH, TEXT_FILE_NAME)
        root = TEST_INPUT_DIR_PATH
        mean_path = os.path.join(TEST_INPUT_DIR_PATH, MEAN_NAME)
        mean = np.load(mean_path)
        self.assertTrue(mean.shape == (3, 256, 256))
        crop_size = IN_SIZE
        preprocessor = data_preprocessor.DataPreprocessor(file_path, root, mean, crop_size)
        answer_labels = [0, 1, 2, 3]
        self.assertTrue(len(preprocessor) == 4)
        dataset = preprocessor.base
        for answer_label, (image, label) in zip(answer_labels, dataset):
            self.assertTrue(answer_label == label)
            self.assertTrue(image.shape == (3, 256, 256))

    def test_get_sample(self):
        file_path = os.path.join(TEST_INPUT_DIR_PATH, TEXT_FILE_NAME)
        root = TEST_INPUT_DIR_PATH
        mean_path = os.path.join(TEST_INPUT_DIR_PATH, MEAN_NAME)
        mean = np.load(mean_path)
        crop_size = IN_SIZE
        preprocessor = data_preprocessor.DataPreprocessor(file_path, root, mean, crop_size, random=False, is_scaled=False)

        self.assertTrue(len(preprocessor) == 4)
        answer_labels = [0, 1, 2, 3]
        answer_images = [int(item) - MEAN_VALUE for item in FILE_NAMES] * 2
        for i, (image, label) in enumerate(preprocessor):
            self.assertTrue(answer_labels[i] == label)
            self.assertTrue(image.shape == (3, IN_SIZE, IN_SIZE))
            self.assertTrue(np.all(np.abs(image - answer_images[i]) < EPSILON))

    def test_get_sample_with_scaling(self):
        file_path = os.path.join(TEST_INPUT_DIR_PATH, TEXT_FILE_NAME)
        root = TEST_INPUT_DIR_PATH
        mean_path = os.path.join(TEST_INPUT_DIR_PATH, MEAN_NAME)
        mean = np.load(mean_path)
        crop_size = IN_SIZE
        preprocessor = data_preprocessor.DataPreprocessor(file_path, root, mean, crop_size, random=False, is_scaled=True)

        self.assertTrue(len(preprocessor) == 4)
        answer_labels = [0, 1, 2, 3]
        answer_images = [(int(item) - MEAN_VALUE) / 255.0 for item in FILE_NAMES] * 2
        for i, (image, label) in enumerate(preprocessor):
            self.assertTrue(answer_labels[i] == label)
            self.assertTrue(image.shape == (3, IN_SIZE, IN_SIZE))
            self.assertTrue(np.all(np.abs(image - answer_images[i]) < EPSILON))

    def tearDown(self):
        if os.path.isdir(TEST_INPUT_DIR_PATH):
            shutil.rmtree(TEST_INPUT_DIR_PATH)
        if os.path.isdir(TEST_OUTPUT_DIR_PATH):
            shutil.rmtree(TEST_OUTPUT_DIR_PATH)


if __name__ == "__main__":
    unittest.main()
