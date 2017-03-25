#!/usr/bin/env python
# coding:utf-8

# import chainer
# from chainer import Variable
from modified_reference_caffenet import *  # noqa
from copy_model import *  # noqa
import cPickle
import os
from data_loader import *  # noqa
# import sys
import argparse
from data_preprocessor import DataPreprocessor
from chainer import training

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


def check_path(path):
    if not os.path.exists(initial_model_path):
        raise IOError("{} is not found".format(path))


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--initial_model_path", help="input: set a path to an initial trained model file(.pkl)")
        parser.add_argument("--root_dir_path", help="input: set a path to an training/testing directory")
        parser.add_argument("--training_data_path", help="input: set a path to a training data file(.txt)")
        parser.add_argument("--testing_data_path", help="input: set a path to a testing data file(.txt)")
        parser.add_argument("--mean_image_path", help="input: set a path to a mean image file(.npy)")
        parser.add_argument("--gpu", type=int, default=-1, help="input: GPU ID(negative value indicates CPU")
        parser.add_argument('--loader_job', type=int, default=2, help='input: number of parallel data loading processes')
        parser.add_argument('--batch_size', type=int, default=32, help='input: learning minibatch size')
        parser.add_argument('--test_batch_size', type=int, default=250, help='input: testing minibatch size')
        parser.add_argument('--epoch', type=int, default=10, help='input: number of epochs to train')
        parser.add_argument('--out_dir_path', default='result', help='output: set a path to output directory')

        args = parser.parse_args()

        # get paths
        initial_model_path = args.initial_model_path
        root_dir_path = args.root_dir_path
        training_data_path = args.training_data_path
        testing_data_path = args.testing_data_path
        mean_image_path = args.mean_image_path
        out_dir_path = args.out_dir_path

        # check paths
        check_path(initial_model_path)
        check_path(root_dir_path)
        check_path(training_data_path)
        check_path(testing_data_path)
        check_path(mean_image_path)
        check_path(out_dir_path)

        print("# _/_/_/ load model _/_/_/")

        # load an original Caffe model
        original_model = cPickle.load(open(initial_model_path))

        # load a new model to be fine-tuned
        modified_model = ModifiedReferenceCaffeNet()

        # copy W/b from the original model to the new one
        copy_model(original_model, modified_model)

        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()  # make the GPU current
            modified_model.to_gpu()

        print("# _/_/_/ load dataset _/_/_/")

        in_size = ModifiedReferenceCaffeNet.IN_SIZE
        mean = np.load(mean_image_path)
        train = DataPreprocessor(training_data_path, root_dir_path, mean, in_size, random=True, is_scaled=True)
        test = DataPreprocessor(testing_data_path, root_dir_path, mean, in_size, random=False, is_scaled=True)

        train_iter = chainer.iterators.MultiprocessIterator(train, args.batch_size, n_processes=args.loader_job)
        test_iter = chainer.iterators.MultiprocessIterator(test, args.test_batch_size, repeat=False, n_processes=args.loader_job)

        print("# _/_/_/ set up an optimizer _/_/_/")

        optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
        optimizer.setup(modified_model)

        print("# _/_/_/ set up a trainer _/_/_/")

        updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
        trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out_dir_path)

        val_interval = (10 if args.test else 100000), 'iteration'
        log_interval = (10 if args.test else 1000), 'iteration'

        # trainer.extend(TestModeEvaluator(val_iter, model, device=args.gpu),
        #                trigger=val_interval)
        # trainer.extend(extensions.dump_graph('main/loss'))
        # trainer.extend(extensions.snapshot(), trigger=val_interval)
        # trainer.extend(extensions.snapshot_object(
        #     model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
        # # Be careful to pass the interval directly to LogReport
        # # (it determines when to emit log rather than when to read observations)
        # trainer.extend(extensions.LogReport(trigger=log_interval))
        # trainer.extend(extensions.observe_lr(), trigger=log_interval)
        # trainer.extend(extensions.PrintReport([
        #     'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        #     'main/accuracy', 'validation/main/accuracy', 'lr'
        # ]), trigger=log_interval)
        # trainer.extend(extensions.ProgressBar(update_interval=10))

        # if args.resume:
        #     chainer.serializers.load_npz(args.resume, trainer)

        # trainer.run()

    except IOError, e:
        print(e)
