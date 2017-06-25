#!/usr/bin/env python
# coding:utf-8

import unittest
from data_preprocessor_for_devise import DataPreprocessorForDevise
# import os
import numpy as np
# import chainer
import sys
sys.path.append('../visual')
from modified_reference_caffenet import *  # noqa


class TestDataPreprocessorForDevise(unittest.TestCase):

    # def construct_instance(self):
    #     root_dir_path = '/home/ubuntu/data/devise/selected_images_256_greater_than_200_images'
    #     training_path = os.path.join(root_dir_path, 'train_valid_selected_.txt')
    #     mean_image_path = '/home/ubuntu/libs/caffe-master/python/caffe/imagenet/ilsvrc_2012_mean.npy'
    #     model_path = '/home/ubuntu/results/devise/20170517-07-20/model_iter_11453'
    #     word2vec_model_path = '/home/ubuntu/results/word2vec/word2vec.model'
    #     class_size = 99
    #     crop_size = 227
    #     mean = np.load(mean_image_path)
    #     gpu = 0

    #     preprocessor = DataPreprocessorForDevise(
    #         training_path,
    #         model_path,
    #         word2vec_model_path,
    #         class_size,
    #         root_dir_path,
    #         mean,
    #         crop_size,
    #         gpu,
    #         random=False,
    #         is_scaled=True
    #     )
    #     return preprocessor

    # def test_init(self):
    #     preprocessor = self.construct_instance()
    #     self.assertTrue(preprocessor is not None)

    # def test_len(self):
    #     preprocessor = self.construct_instance()
    #     self.assertTrue(len(preprocessor) == 49628)

    def make_dummy_image(self):
        # batch, channel, row, col
        return np.random.randint(0, 256, (1, 3, 227, 227)).astype('f')

    # def make_dummy_image_2(self):
    #     # batch, channel, row, col
    #     return np.random.randint(0, 256, (3, 227, 227)).astype('f')

    # def test_load_model(self):
    #     class_size = 99
    #     gpu = 0
    #     model_path = '/home/ubuntu/results/devise/20170517-07-20/model_iter_11453'
    #     model = DataPreprocessorForDevise.load_model(model_path, class_size, gpu)

    #     dummy_image = self.make_dummy_image()
    #     dummy_image = chainer.cuda.to_gpu(dummy_image)
    #     feature, softmax = model(dummy_image, None)
    #     self.assertTrue(feature.shape == (1, 4096))

    #     original_model = ModifiedReferenceCaffeNet(class_size)
    #     chainer.serializers.load_npz(model_path, original_model)
    #     original_model.select_phase('predict')
    #     original_model.to_gpu()
    #     answer_softmax = original_model(dummy_image, None)

    #     self.assertTrue(np.all(chainer.cuda.to_cpu(answer_softmax.data) == chainer.cuda.to_cpu(softmax.data)))

    def test_load_word2vec_model(self):
        word2vec_model_path = '/home/ubuntu/results/word2vec/word2vec.model'
        (word2index, index2word, word2vec_w) = DataPreprocessorForDevise.load_word2vec_model(word2vec_model_path)
        self.assertTrue(word2index['wringer'] == 223093)
        self.assertTrue(index2word[223093] == 'wringer')

    # def test_convert_to_feature(self):
    #     preprocessor = self.construct_instance()
    #     dummy_image = self.make_dummy_image_2()
    #     dummy_image = chainer.cuda.to_gpu(dummy_image)
    #     f = preprocessor.convert_to_feature(dummy_image)
    #     self.assertTrue(f.shape == (1, 4096))

    # def test_convert_to_word_vector(self):
    #     preprocessor = self.construct_instance()
    #     v = preprocessor.convert_to_word_vector(98)  # 'wringer')
    #     self.assertTrue(v.shape == (200,))
    #     a = np.linalg.norm(v)
    #     self.assertTrue(abs(a - 1.0) < 1.0e-05)


if __name__ == '__main__':
    unittest.main()
