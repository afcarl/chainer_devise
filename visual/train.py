#!/usr/bin/env python
# coding:utf-8

from modified_reference_caffenet import *  # noqa
from copy_model import *  # noqa
from chainer.training import extensions
import cPickle
import os
import argparse
from data_preprocessor import DataPreprocessor
from chainer import training
import numpy as np


def check_path(path):
    if not os.path.exists(initial_model_path):
        raise IOError("{} is not found".format(path))


def load_labels(path):
    return sum([1 for _ in open(path)])


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.select_phase('test')
        ret = super(TestModeEvaluator, self).evaluate()
        model.select_phase('train')
        return ret


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--initial_model_path", help="input: set a path to an initial trained model file(.pkl)")
        parser.add_argument("--root_dir_path", help="input: set a path to an training/testing directory")
        parser.add_argument("--label_path", help="input: set a path to a label file")
        parser.add_argument("--training_data_path", help="input: set a path to a training data file(.txt)")
        parser.add_argument("--testing_data_path", help="input: set a path to a testing data file(.txt)")
        parser.add_argument("--mean_image_path", help="input: set a path to a mean image file(.npy)")
        parser.add_argument("--gpu", type=int, default=-1, help="input: GPU ID(negative value indicates CPU")
        parser.add_argument('--loader_job', type=int, default=2,
                            help='input: number of parallel data loading processes')
        parser.add_argument('--batch_size', type=int, default=32, help='input: learning minibatch size')
        parser.add_argument('--test_batch_size', type=int, default=250, help='input: testing minibatch size')
        parser.add_argument('--epoch', type=int, default=10, help='input: number of epochs to train')
        parser.add_argument('--out_dir_path', default='result', help='output: set a path to output directory')
        parser.add_argument('--test', action='store_true', default=False,
                            help='option: test mode if this flag is set(default: False)')
        parser.add_argument('--resume', default='', help='option: initialize the trainer from given file')
        parser.add_argument('--log_interval', type=int, help='input: test interval')
        parser.add_argument('--model_epoch', type=int, default=1, help='input: epoch to save model')

        args = parser.parse_args()

        # get paths
        initial_model_path = args.initial_model_path
        root_dir_path = args.root_dir_path
        label_path = args.label_path
        training_data_path = args.training_data_path
        testing_data_path = args.testing_data_path
        mean_image_path = args.mean_image_path
        out_dir_path = args.out_dir_path

        # check paths
        check_path(initial_model_path)
        check_path(root_dir_path)
        check_path(label_path)
        check_path(training_data_path)
        check_path(testing_data_path)
        check_path(mean_image_path)
        check_path(out_dir_path)

        print("# _/_/_/ load model _/_/_/")

        # load an original Caffe model
        original_model = cPickle.load(open(initial_model_path))

        # load a new model to be fine-tuned
        class_size = load_labels(label_path)
        modified_model = ModifiedReferenceCaffeNet(class_size)

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
        test_iter = chainer.iterators.MultiprocessIterator(test, args.test_batch_size, repeat=False,
                                                           n_processes=args.loader_job)

        print("# _/_/_/ set up an optimizer _/_/_/")

        # optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(modified_model)

        print("# _/_/_/ set up a trainer _/_/_/")

        updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
        trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out_dir_path)

        log_interval = (10 if args.test else args.log_interval), 'iteration'
        model_epoch = (1 if args.test else args.model_epoch), 'epoch'

        trainer.extend(TestModeEvaluator(test_iter, modified_model, device=args.gpu), trigger=log_interval)
        trainer.extend(extensions.dump_graph('main/loss'))  # yield cg.dot
        trainer.extend(extensions.snapshot(), trigger=model_epoch)  # save a trainer for resuming training
        trainer.extend(extensions.snapshot_object(modified_model, 'model_iter_{.updater.iteration}'),
                       trigger=model_epoch)  # save a modified model

        # Be careful to pass the interval directly to LogReport
        # (it determines when to emit log rather than when to read observations)
        trainer.extend(extensions.LogReport(trigger=log_interval))  # yield 'log'
        trainer.extend(extensions.observe_lr(), trigger=log_interval)

        # Save two plot images to the result dir
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'validation/main/loss'], 'iteration', trigger=log_interval, file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'], 'iteration', trigger=log_interval, file_name='accuracy.png'))

        trainer.extend(
            extensions.PrintReport(
                ['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'main/accuracy',
                 'validation/main/accuracy', 'lr']),
            trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))

        if args.resume:
            chainer.serializers.load_npz(args.resume, trainer)

        trainer.run()
    except IOError, e:
        print(e)
