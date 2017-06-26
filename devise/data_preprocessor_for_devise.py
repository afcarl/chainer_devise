#!/usr/bin/env python
# coding:utf-8

import sys
sys.path.append('../visual')
from image_cropper import *  # noqa
from modified_reference_caffenet import *  # noqa
from modified_reference_caffenet_with_extractor import *  # noqa
from chainer import dataset  # noqa
from chainer import datasets  # noqa
import random  # noqa
from copy_model import *  # noqa
# import numpy as np


class DataPreprocessorForDevise(dataset.DatasetMixin):

    # test ok
    def __init__(
        self,
        path,
        model,
        (word2index, label2word, word2vec_w),
        class_size,
        root,
        mean,
        crop_size,
        gpu,
        random=True,
        is_scaled=True
    ):
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
        self.model = model
        self.word2index, self.label2word, self.word2vec_w = (word2index, label2word, word2vec_w)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random
        self.is_scaled = is_scaled

    # test ok
    @staticmethod
    def load_model(model_path, class_size, gpu):
        original_model = ModifiedReferenceCaffeNet(class_size)
        chainer.serializers.load_npz(model_path, original_model)
        model = ModifiedReferenceCaffeNetWithExtractor(class_size)
        copy_model(original_model, model)
        model.select_phase('extractor')
        if gpu >= 0:
            chainer.cuda.get_device(gpu).use()  # make the GPU current
            model.to_gpu()
        return model

    # test ok
    @staticmethod
    def load_word2vec_model(word2vec_model_path):
        with open(word2vec_model_path, 'r') as f:
            ss = f.readline().split()
            n_vocab, n_units = int(ss[0]), int(ss[1])
            word2index = {}
            w = np.empty((n_vocab, n_units), dtype=np.float32)
            for i, line in enumerate(f):
                ss = line.split()
                assert len(ss) == n_units + 1
                word = ss[0]
                word2index[word] = i
                w[i] = np.array([float(s) for s in ss[1:]], dtype=np.float32)

        s = np.sqrt((w * w).sum(1))
        w /= s.reshape((s.shape[0], 1))  # normalize
        return word2index, w

    # test ok
    @staticmethod
    def load_labels(label_path):
        label2word = {}
        for line in open(label_path):
            line = line.strip()
            if line == '':
                continue
            tokens = line.split()
            if len(tokens) != 2:
                continue
            label2word[int(tokens[1])] = tokens[0]
        return label2word

    # test ok
    def __len__(self):
        return len(self.base)

    # test ok
    def convert_to_feature(self, image):
        """
        @param image an image instance
        @return feature vector
        """
        x = image[np.newaxis]
        return self.model(x, None)[0]

    # test ok
    def convert_to_word_vector(self, label):
        """
        @param label a label
        @return word vector
        """
        word = self.label2word[label]
        index = self.word2index[word]
        return self.word2vec_w[index]

    def get_example(self, i):
        """
        This method reads the i-th pair of (image, label) and return a pair of (feature_vector, word_vector).
        It applies the following preprocesses to the image:
          - Cropping (random or center rectangular)
          - Scaling to [0, 1] value
        and the following preprocesses to the label:
          - Nomalizing
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
