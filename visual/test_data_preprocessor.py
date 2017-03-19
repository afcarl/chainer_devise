#!/usr/bin/env python
# coding:utf-8
import cv2
import unittest
from data_preprocessor import *  # noqa
import shutil

TEST_INPUT_DIR_PATH = "../unittest_input"
TEST_OUTPUT_DIR_PATH = "../unittest_output"
SUB_DIR_NAMES = ["hoge", "foo"]
MEAN_NAME = "meang.npy"
MEAN_VALUE = 3
FILE_NAMES = ["1", "2"]
CROPPING_SIZE = 227
EPSILON = 1.0e-08


class TestDataPreprocessor(unittest.TestCase):

    @staticmethod
    def make_dirs():
        sub_dir_paths = [os.path.join(TEST_INPUT_DIR_PATH, name) for name in SUB_DIR_NAMES]
        for sub_dir_path in sub_dir_paths:
            if not os.path.isdir(sub_dir_path):
                os.makedirs(sub_dir_path)
        return sub_dir_paths

    @staticmethod
    def make_test_image(value):
        image = np.zeros((256, 256, 3))
        image[14:241, 14:241, :] = value
        return image

    def setUp(self):
        # make input dirs and input files
        sub_dir_paths = TestDataPreprocessor.make_dirs()
        for sub_dir_path in sub_dir_paths:
            for file_name in FILE_NAMES:
                image = TestDataPreprocessor.make_test_image(int(file_name))
                image_path = os.path.join(sub_dir_path, file_name + ".png")
                cv2.imwrite(image_path, image)
        mean_image = TestDataPreprocessor.make_test_image(MEAN_VALUE).transpose(2, 0, 1)
        mean_image_path = os.path.join(TEST_INPUT_DIR_PATH, MEAN_NAME)
        np.save(mean_image_path, mean_image)

    def test_save_images(self):
        mean_image_path = os.path.join(TEST_INPUT_DIR_PATH, MEAN_NAME)
        preprocessor = DataPreprocessor(CROPPING_SIZE, mean_image_path)
        if not os.path.isdir(TEST_OUTPUT_DIR_PATH):
            os.mkdir(TEST_OUTPUT_DIR_PATH)

        preprocessor.save_images(TEST_INPUT_DIR_PATH, TEST_OUTPUT_DIR_PATH)

        for dirname in os.listdir(TEST_OUTPUT_DIR_PATH):
            path = os.path.join(TEST_OUTPUT_DIR_PATH, dirname)
            self.check_images(path)

    def test_image_generator(self):
        c = 0
        names = []
        input_dir_path = os.path.join(TEST_INPUT_DIR_PATH, SUB_DIR_NAMES[0])
        for image, name in DataPreprocessor.image_generator(input_dir_path, is_verbose=False):
            c += 1
            names.append(name)
        self.assertTrue(c == 2)
        self.assertTrue(FILE_NAMES == names)

    def test_init(self):
        mean_image_path = os.path.join(TEST_INPUT_DIR_PATH, MEAN_NAME)
        preprocessor = DataPreprocessor(CROPPING_SIZE, mean_image_path)
        cropped_mean_image = preprocessor.cropped_mean_image
        self.assertTrue((3, CROPPING_SIZE, CROPPING_SIZE) == cropped_mean_image.shape)
        self.assertTrue(np.all(np.abs(cropped_mean_image - MEAN_VALUE) < EPSILON))

    def check_images(self, output_dir_path):
        answers = [int(item) - MEAN_VALUE for item in FILE_NAMES]
        for i, item in enumerate(os.listdir(output_dir_path)):
            path = os.path.join(output_dir_path, item)
            image = np.load(path)
            self.assertTrue(np.all(np.abs(image - answers[i]) < EPSILON))

    def test_save_to_one_directory(self):
        mean_image_path = os.path.join(TEST_INPUT_DIR_PATH, MEAN_NAME)
        preprocessor = DataPreprocessor(CROPPING_SIZE, mean_image_path)
        input_dir_path = os.path.join(TEST_INPUT_DIR_PATH, SUB_DIR_NAMES[0])
        output_dir_path = os.path.join(TEST_OUTPUT_DIR_PATH, SUB_DIR_NAMES[0])
        if not os.path.isdir(output_dir_path):
            os.makedirs(output_dir_path)
        preprocessor.save_to_one_directory(input_dir_path, output_dir_path)

        # check values
        self.check_images(output_dir_path)

    def tearDown(self):
        if os.path.isdir(TEST_INPUT_DIR_PATH):
            shutil.rmtree(TEST_INPUT_DIR_PATH)
        if os.path.isdir(TEST_OUTPUT_DIR_PATH):
            shutil.rmtree(TEST_OUTPUT_DIR_PATH)


if __name__ == "__main__":
    unittest.main()
