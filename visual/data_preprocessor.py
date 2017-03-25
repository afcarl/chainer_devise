#!/usr/bin/env python
# coding:utf-8

from image_cropper import *  # noqa
from chainer import dataset
from chainer import datasets
import random  # noqa


class DataPreprocessor(dataset.DatasetMixin):

    # test ok
    def __init__(self, path, root, mean, crop_size, random=True, is_scaled=True):
        self.base = datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random
        self.is_scaled = is_scaled

    # test ok
    def __len__(self):
        return len(self.base)

    # test ok
    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #   - Cropping (random or center rectangular)
        #   - Random flip
        #   - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2

        # Crop an image
        bottom = top + crop_size
        right = left + crop_size
        image = image[:, top:bottom, left:right]

        # Subtract a mean image
        image -= self.mean[:, top:bottom, left:right]

        # If necessary, scale an image
        if self.is_scaled:
            image *= (1.0 / 255.0)
        return image, label
