#!/usr/bin/env python
# coding:utf-8
from image_cropper import *  # noqa
import unittest
import numpy as np
import os
import shutil

TEST_IMAGE_DIR_PATH = "../unittest_files"
TEST_IMAGE_PATH = os.path.join(TEST_IMAGE_DIR_PATH, "test.png")
INSIZE = 227
EPSILON = 1.0e-8


class ImageCropperTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(ImageCropperTest, self).__init__(*args, **kwargs)
        self.image = ImageCropperTest.make_test_image()
        if not os.path.isdir(TEST_IMAGE_DIR_PATH):
            os.mkdir(TEST_IMAGE_DIR_PATH)
        cv2.imwrite(TEST_IMAGE_PATH, self.image)

    @staticmethod
    def make_test_image():
        image = np.zeros((256, 256, 3))
        image[14:241, 14:241, :] = 2
        return image

    def test_crop_center(self):
        cropper = ImageCropper(INSIZE)
        cropped_image = cropper.crop_center(TEST_IMAGE_PATH, is_scaled=False)
        self.assertTrue(cropped_image.shape == (3, INSIZE, INSIZE))
        self.assertTrue(np.all(cropped_image == 2))

    def test_crop_center_image(self):
        image = self.image.transpose(2, 0, 1)
        cropper = ImageCropper(INSIZE)
        cropped_image = cropper.crop_center_image(image, is_scaled=False)
        self.assertTrue((3, INSIZE, INSIZE) == cropped_image.shape)
        self.assertTrue(np.all(cropped_image == 2))

    def tearDown(self):
        if os.path.isdir(TEST_IMAGE_DIR_PATH):
            shutil.rmtree(TEST_IMAGE_DIR_PATH)


if __name__ == "__main__":
    unittest.main()
