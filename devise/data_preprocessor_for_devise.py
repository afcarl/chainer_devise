#!/usr/bin/env python
# coding:utf-8

import sys
sys.path.append('../visual')
from image_cropper import *  # noqa
from modified_reference_caffenet import *  # noqa
from chainer import dataset  # noqa
from chainer import datasets  # noqa
import random  # noqa


class DataPreprocessorForDevise(dataset.DatasetMixin):

    # test ok
    def __init__(self, path, model_path, class_size, root, mean, crop_size, gpu, random=True, is_scaled=True):
        """
        @param path a path to a training/testing data file
        @param root a path to a training/teting directory
        @param mean a np.array instance of an average image
        @param crop_size
        @param random True if random selection is needed.
        @param is_scaled True if a scaling is needed. This value must be the same as training procedure.
        """
        self.base = datasets.LabeledImageDataset(path, root)
        self.gpu = gpu
        self.model = self.load_model(model_path, class_size, gpu)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random
        self.is_scaled = is_scaled

    def load_model(self, model_path, class_size, gpu):
        model = ModifiedReferenceCaffeNet(class_size)
        chainer.serializers.load_npz(model_path, model)
        model.select_phase('predict')
        if gpu >= 0:
            chainer.cuda.get_device(gpu).use()  # make the GPU current
            model.to_gpu()
        return model

    # test ok
    def __len__(self):
        return len(self.base)

    def convert_to_feature(self, image):
        """
        @param image an image instance
        @return feature vector
        """
        x = image[np.newaxis]
        self.model(x, None)
        return image

    def convert_to_word_vector(self, label):
        """
        @param label a label
        @return word vector
        """
        return label

    def get_example(self, i):
        """
        This method reads the i-th pair of (image, label) and return a pair of (feature_vector, word_vector).
        It applies following preprocesses to the former pair:
          - Cropping (random or center rectangular)
          - Scaling to [0, 1] value
        """
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
        return self.convert_to_feature(image), self.convert_to_word_vector(label)
