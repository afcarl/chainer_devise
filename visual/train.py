#!/usr/bin/env python
# coding:utf-8

# import chainer
# from chainer import Variable
from modified_reference_caffenet import *  # noqa
from copy_model import *  # noqa
import cPickle as pickle
# import os
from data_loader import *  # noqa
# import sys
import argparse

# _/_/_/ paths _/_/_/
# PICKLE_PATH = "/home/ubuntu/data/models/chainer/bvlc_reference_caffenet/bvlc_reference_caffenet-2017-01-08.pkl"
# PICKLE_DUMP_PATH = "/home/ubuntu/results/devise/trainded_visual_model.pkl"
# MEAN_IMAGE_PATH = "/home/ubuntu/libs/caffe-master/python/caffe/imagenet/ilsvrc_2012_mean.npy"


# _/_/_/ training parameters _/_/_/
# LEARNIN_RATE = 0.01
# BATCH_SIZE = 20
# EPOCHS = 100
# DECAY_FACTOR = 0.97


# _/_/_/ dataset _/_/_/
# DATA_ROOT_DIR_PATH = "/home/ubuntu/data/selected_images_256"
# TRAINING_DATA_PATH = os.path.join(DATA_ROOT_DIR_PATH, "train_valid.txt")
# TESTING_DATA_PATH = os.path.join(DATA_ROOT_DIR_PATH, "test.txt")


# def test(x_test, y_test, model):
#     sum_accuracy = 0
#     sum_loss = 0
#     test_data_size = len(x_test)
#     for i in range(0, test_data_size, BATCH_SIZE):
#         x = chainer.Variable(chainer.cuda.to_gpu(x_test[i: i + BATCH_SIZE]))
#         t = chainer.Variable(chainer.cuda.to_gpu(y_test[i: i + BATCH_SIZE]))
#         loss = model(x, t)
#         sum_loss += loss.data * BATCH_SIZE
#         sum_accuracy += model.accuracy.data * BATCH_SIZE
#
#     print("test mean loss {a}, accuracy {b}".format(a=sum_loss / test_data_size, b=sum_accuracy / test_data_size))
#     sys.stdout.flush()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--initial_model_path", help="input: set a path to an initial trained model file(.pkl)")
        parser.add_argument("--training_data_path", help="input: set a path to a training data file(.txt)")
        parser.add_argument("--testing_data_path", help="input: set a path to a testing data file(.txt)")
        parser.add_argument("--mean_image_path", help="input: set a path to a mean image file(.npy)")

        args = parser.parse_args()
        initial_model_path = args.initial_model_path
        training_data_path = args.training_data_path
        testing_data_path = args.testing_data_path
        mean_image_path = args.mean_image_path

        print("# _/_/_/ load dataset _/_/_/")

        data_loader = DataLoader()
        x_train, y_train = data_loader.load_with_subtraction_of_mean(
            training_data_path,
            mean_image_path,
            ModifiedReferenceCaffeNet.IN_SIZE
        )

        x_test, y_test = data_loader.load_with_subtraction_of_mean(
            testing_data_path,
            mean_image_path,
            ModifiedReferenceCaffeNet.IN_SIZE
        )

        print("train x.shape: {s}, x.dtype: {d}".format(s=x_train.shape, d=x_train.dtype))
        print("train y.shape: {s}, y.dtype: {d}".format(s=y_train.shape, d=y_train.dtype))

        print("test x.shape: {s}, x.dtype: {d}".format(s=x_test.shape, d=x_test.dtype))
        print("test y.shape: {s}, y.dtype: {d}".format(s=y_test.shape, d=y_test.dtype))
        sys.stdout.flush()

        print("# _/_/_/ load model _/_/_/")

        # load an original Caffe model
        original_model = pickle.load(open(initial_model_path))

        # load a new model to be fine-tuned
        modified_model = ModifiedReferenceCaffeNet()

        # copy W/b from the original model to the new one
        copy_model(original_model, modified_model)

        # _/_/_/ setup _/_/_/

        # model = modified_model.to_gpu()
        # optimizer = chainer.optimizers.SGD(LEARNIN_RATE)
        # optimizer.setup(model)

        # # _/_/_/ training _/_/_/

        # train_data_size = len(x_train)

        # for epoch in range(1, EPOCHS + 1):
        #     print("epoch %d" % epoch)
        #     sys.stdout.flush()
        #     indices = np.random.permutation(train_data_size)
        #     sum_accuracy = 0
        #     sum_loss = 0

        #     for i in range(0, train_data_size, BATCH_SIZE):
        #         r = indices[i: i + BATCH_SIZE]
        #         x = Variable(chainer.cuda.to_gpu(x_train[r]))
        #         y = Variable(chainer.cuda.to_gpu(y_train[r]))
        #         model.zerograds()
        #         loss = model(x, y)
        #         loss.backward()
        #         optimizer.update()

        #         sum_loss += loss.data * BATCH_SIZE
        #         sum_accuracy += model.accuracy.data * BATCH_SIZE

        #     print("train mean loss {a}, accuracy {b}".format(a=sum_loss / train_data_size, b=sum_accuracy / train_data_size))
        #     sys.stdout.flush()
        #     test(x_test, y_test, model)
        #     optimizer.lr *= DECAY_FACTOR

        # # _/_/_/ testing _/_/_/

        # test(x_test, y_test, model)

        # # _/_/_/ saving _/_/_/

        # pickle.dump(model, open(PICKLE_DUMP_PATH, "wb"))
    except IOError, e:
        print(e)
