#!/usr/bin/env python
# coding:utf-8

import unittest
from data_preprocessor_for_devise import DataPreprocessorForDevise
import os
import numpy as np
import chainer.cuda
import sys
sys.path.append('../visual')
from modified_reference_caffenet import *  # noqa


class TestDataPreprocessorForDevise(unittest.TestCase):
    preprocessor = None
    index2word = None

    @classmethod
    def setUpClass(cls):
        root_dir_path = '/home/ubuntu/data/devise/selected_images_256_greater_than_200_images'
        training_path = os.path.join(root_dir_path, 'train_valid_selected_.txt')
        mean_image_path = '/home/ubuntu/libs/caffe-master/python/caffe/imagenet/ilsvrc_2012_mean.npy'
        model_path = '/home/ubuntu/results/devise/20170517-07-20/model_iter_11453'
        word2vec_model_path = '/home/ubuntu/results/word2vec/word2vec.model'
        label_path = '/home/ubuntu/data/devise/selected_images_256_greater_than_200_images/label_selected.txt'

        class_size = 99
        crop_size = 227
        mean = np.load(mean_image_path)
        gpu = 0

        model = DataPreprocessorForDevise.load_model(model_path, class_size, gpu)
        word2index, index2word, word2vec_w = DataPreprocessorForDevise.load_word2vec_model(word2vec_model_path)
        label2word = DataPreprocessorForDevise.load_labels(label_path)

        cls.preprocessor = DataPreprocessorForDevise(
            training_path,
            model,
            (word2index, label2word, word2vec_w),
            class_size,
            root_dir_path,
            mean,
            crop_size,
            gpu,
            random=False,
            is_scaled=True
        )
        cls.index2word = index2word

    def test_init(self):
        self.assertTrue(TestDataPreprocessorForDevise.preprocessor is not None)

    def test_len(self):
        self.assertTrue(len(TestDataPreprocessorForDevise.preprocessor) == 49628)

    def make_dummy_image(self):
        # batch, channel, row, col
        return np.random.randint(0, 256, (1, 3, 227, 227)).astype('f')

    def make_dummy_image_2(self):
        # batch, channel, row, col
        return np.random.randint(0, 256, (3, 227, 227)).astype('f')

    def test_load_model(self):
        model = TestDataPreprocessorForDevise.preprocessor.model

        dummy_image = self.make_dummy_image()
        dummy_image = chainer.cuda.to_gpu(dummy_image)
        feature, softmax = model(dummy_image, None)
        self.assertTrue(feature.shape == (1, 4096))

        model_path = '/home/ubuntu/results/devise/20170517-07-20/model_iter_11453'
        class_size = 99
        original_model = ModifiedReferenceCaffeNet(class_size)
        chainer.serializers.load_npz(model_path, original_model)
        original_model.select_phase('predict')
        original_model.to_gpu()
        answer_softmax = original_model(dummy_image, None)

        self.assertTrue(np.all(chainer.cuda.to_cpu(answer_softmax.data) == chainer.cuda.to_cpu(softmax.data)))

    def test_load_word2vec_model(self):
        word2vec_model_path = '/home/ubuntu/results/word2vec/word2vec.model'
        (word2index,  _, word2vec_w) = DataPreprocessorForDevise.load_word2vec_model(word2vec_model_path)
        self.assertTrue(word2index['wringer'] == 223093)

    def test_convert_to_feature(self):
        dummy_image = self.make_dummy_image_2()
        f = TestDataPreprocessorForDevise.preprocessor.convert_to_feature(dummy_image)
        self.assertTrue(f.shape == (1, 4096))

    def test_convert_to_word_vector(self):
        v = TestDataPreprocessorForDevise.preprocessor.convert_to_word_vector(98)
        self.assertTrue(v.shape == (200,))
        a = np.linalg.norm(v)
        self.assertTrue(abs(a - 1.0) < 1.0e-05)

    def test_load_labels(self):
        path = '/home/ubuntu/data/devise/selected_images_256_greater_than_200_images/label_selected.txt'
        label2word = DataPreprocessorForDevise.load_labels(path)
        self.assertTrue(label2word[98] == 'wringer')

    def test_get_example(self):
        preprocessor = TestDataPreprocessorForDevise.preprocessor
        feature, w2v = preprocessor.get_example(0)
        self.assertTrue(feature.shape == (1, 4096))
        self.assertTrue(w2v.shape == (200, 1 + preprocessor.n_similarities))

    def test_iter(self):
        preprocessor = TestDataPreprocessorForDevise.preprocessor
        batch_size = 3
        iterator = chainer.iterators.SerialIterator(preprocessor, batch_size, shuffle=False)
        rs = iterator.next()
        self.assertTrue(type(rs) == list)
        self.assertTrue(len(rs) == batch_size)
        for r in rs:
            self.assertTrue(len(r) == 2)
            self.assertTrue(type(r[0]) == chainer.cuda.ndarray)
            self.assertTrue(type(r[1]) == np.ndarray)
            self.assertTrue(r[0].shape == (1, 4096))
            self.assertTrue(r[1].shape == (200, 1 + preprocessor.n_similarities))

    def test_word_searcher(self):
        preprocessor = TestDataPreprocessorForDevise.preprocessor
        word_searcher = preprocessor.word_searcher
        similar_indices = list(word_searcher.similarity_generator('tokyo'))
        answers = [(396750, 0.88120759), (619085, 0.82389867), (502362, 0.81960642), (98620, 0.794792), (515730, 0.78090179)]
        # print(similar_indices)
        for (similar_index, answer) in zip(similar_indices, answers):
            self.assertTrue(similar_index[0] == answer[0])
            self.assertAlmostEqual(similar_index[1], answer[1], delta=1.0e-5)

    def test_find_similar_words(self):
        preprocessor = TestDataPreprocessorForDevise.preprocessor
        # word = 'aircraft'
        label = 1  # a label used in visual model
        similar_indices = preprocessor.find_similar_indices(label)
        similar_words = [TestDataPreprocessorForDevise.index2word[index] for index in similar_indices]
        answers = ['airframe', 'airliner', 'rotorcraft', 'airframes', 'airliners']
        self.assertTrue(answers == similar_words)

    def test_convert_to_word_vectors(self):
        preprocessor = TestDataPreprocessorForDevise.preprocessor
        label = 1
        vecs = preprocessor.convert_to_word_vectors(label)
        self.assertTrue(vecs.shape == (200, 1 + preprocessor.n_similarities))


if __name__ == '__main__':
    unittest.main()
