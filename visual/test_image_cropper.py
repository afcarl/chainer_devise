#!/usr/bin/env python
# coding:utf-8
from image_cropper import *  # noqa
import unittest
# import subprocess

SAMPLE_IMAGE_PATH = "../unittest_files/image_cropper/11_Pleospora048DFs.png"
SAVING_PATH = "../unittest_files/image_cropper/ans_11_Pleospora048DFs.png"
INSIZE = 227
EPSILON = 1.0e-8


class ImageCropperTest(unittest.TestCase):

    def test_crop_center(self):
        cropper = ImageCropper(INSIZE)
        center_image = cropper.crop_center(SAMPLE_IMAGE_PATH, is_scaled=False)
        self.assertTrue(center_image.shape == (3, INSIZE, INSIZE))
        answer = cv2.imread(SAVING_PATH).transpose(2, 0, 1).astype(np.float32)
        self.assertTrue(np.all(answer == center_image))

    def test_crop_center_image(self):
        image = cv2.imread(SAMPLE_IMAGE_PATH).transpose(2, 0, 1)
        cropper = ImageCropper(INSIZE)
        image = cropper.crop_center_image(image, is_scaled=False)
        self.assertTrue((3, INSIZE, INSIZE) == image.shape)
        answer = cv2.imread(SAVING_PATH).transpose(2, 0, 1).astype(np.float32)
        self.assertTrue(np.all(answer == image))


if __name__ == "__main__":
    unittest.main()
