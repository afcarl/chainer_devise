#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-


import argparse
import os
from devise_in_first_stage import DeviseInFirstStage
from data_preprocessor_for_devise import DataPreprocessorForDevise
import chainer
import numpy as np
from chainer import training
from chainer.training import extensions

VISUAL_FEATURE_SIZE = 4096
WORD2VEC_SIZE = 200


def check_file_path(file_path):
    if not os.path.isfile(file_path):
        raise IOError('invalid file path: {}'.format(file_path))


def check_dir_path(dir_path):
    if not os.path.isdir(dir_path):
        raise IOError('invalid dir path: {}'.format(dir_path))


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        # model = self.get_target('main')
        # model.select_phase('test')
        ret = super(TestModeEvaluator, self).evaluate()
        # model.select_phase('train')
        return ret


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()

        parser.add_argument('--gpu', default=0, type=int, help='INPUT: GPU ID (negative value indicates CPU)')
        parser.add_argument('--root_dir_path',
                            help='INPUT: a path to a training/testing directory')
        parser.add_argument('--training_path',
                            help='INPUT: a path to a file in which all paths needed for training are described')
        parser.add_argument('--testing_path',
                            help='INPUT: a path to a file in which all paths needed for testing are described')
        parser.add_argument('--mean_image_path',
                            help='INPUT: a path to a mean image file')
        parser.add_argument('--label_path',
                            help='INPUT: a path to a label file')
        parser.add_argument('--out_dir_path', help='OUTPUT: a path to an output directory')
        parser.add_argument('--batch_size', default=32, type=int, help='INPUT: minibatch size')
        parser.add_argument('--test_batch_size', type=int, default=25, help='input: testing minibatch size')
        parser.add_argument('--epoch', default=10, type=int, help='INPUT: number of epochs to train')
        parser.add_argument('--log_interval', type=int, help='INPUT: test interval')
        parser.add_argument('--class_size', type=int, help='INPUT: test interval')
        parser.add_argument('--model_epoch', type=int, default=1, help='INPUT: epoch to save model')
        parser.add_argument('--model_path', help='INPUT: a path to a trained model')
        parser.add_argument('--word2vec_model_path', help='INPUT: a path to a trained word2vec model')
        parser.add_argument('--loader_job', type=int, default=2,
                            help='input: number of parallel data loading processes')
        args = parser.parse_args()

        # check paths
        check_file_path(args.training_path)
        check_file_path(args.testing_path)
        check_file_path(args.mean_image_path)
        check_file_path(args.model_path)
        check_file_path(args.label_path)
        check_file_path(args.word2vec_model_path)
        check_dir_path(args.root_dir_path)

        print('# _/_/_/ make an instance of a model _/_/_/')

        model = DeviseInFirstStage(VISUAL_FEATURE_SIZE, WORD2VEC_SIZE)
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()
            model.to_gpu()

        print('# _/_/_/ load dataset _/_/_/')

        visual_model = DataPreprocessorForDevise.load_model(args.model_path, args.class_size, args.gpu)
        word2index, word2vec_w = DataPreprocessorForDevise.load_word2vec_model(args.word2vec_model_path)
        label2word = DataPreprocessorForDevise.load_labels(args.label_path)

        in_size = DeviseInFirstStage.IN_SIZE
        mean = np.load(args.mean_image_path)
        train = DataPreprocessorForDevise(
            args.training_path,
            visual_model,
            (word2index, label2word, word2vec_w),
            args.class_size,
            args.root_dir_path,
            mean,
            in_size,
            args.gpu,
            random=False,
            is_scaled=True)
        test = DataPreprocessorForDevise(
            args.testing_path,
            visual_model,
            (word2index, label2word, word2vec_w),
            args.class_size,
            args.root_dir_path,
            mean,
            in_size,
            args.gpu,
            random=False,
            is_scaled=True)

        train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
        test_iter = chainer.iterators.SerialIterator(test, args.test_batch_size, repeat=False)

        print("# _/_/_/ set up an optimizer _/_/_/")

        optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
        # optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)

        print("# _/_/_/ set up a trainer _/_/_/")

        updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
        trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out_dir_path)

        log_interval = (args.log_interval, 'iteration')
        model_epoch = (args.model_epoch, 'epoch')

        trainer.extend(TestModeEvaluator(test_iter, model, device=args.gpu), trigger=log_interval)
        trainer.extend(extensions.ExponentialShift('lr', 0.97), trigger=(1, 'epoch'))

        trainer.extend(extensions.dump_graph('main/loss'))  # yield cg.dot
        trainer.extend(extensions.snapshot(), trigger=model_epoch)  # save a trainer for resuming training
        trainer.extend(extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
                       trigger=model_epoch)  # save a modified model

        # Be careful to pass the interval directly to LogReport
        # (it determines when to emit log rather than when to read observations)
        trainer.extend(extensions.LogReport(trigger=log_interval))  # yield 'log'
        trainer.extend(extensions.observe_lr(), trigger=log_interval)

        # Save two plot images to the result dir
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'validation/main/loss'], 'iteration', trigger=log_interval, file_name='loss.png'))
        # trainer.extend(
        #     extensions.PlotReport(
        #         ['main/accuracy', 'validation/main/accuracy'], 'iteration', trigger=log_interval, file_name='accuracy.png'))

        # if args.resume:
        #    chainer.serializers.load_npz(args.resume, trainer)
        print("# _/_/_/ run _/_/_/")
        trainer.run()
    except Exception, e:
        print(e)
