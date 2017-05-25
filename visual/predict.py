#!/usr/bin/env python
# coding:utf-8


import argparse
from modified_reference_caffenet import *  # noqa
import collections
from data_preprocessor import DataPreprocessor
import chainer.cuda
import numpy as np
import cupy

PathInfo = collections.namedtuple('PathInfo', ['path', 'label'])
# CLASS_SIZE = 130


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--testing_data_path", help="input: a path to a file in which images to be classified are described")
        parser.add_argument("--root_dir_path", help="input: a path to a directory in which there are images to be classified")
        parser.add_argument("--model_path", help="input: a path to a trained model")
        parser.add_argument("--mean_image_path", help="input: a path to a mean image")
        parser.add_argument("--gpu", type=int, default=-1, help="input: GPU ID(negative value indicates CPU")
        parser.add_argument('--batch_size', type=int, default=32, help='input: minibatch size')
        parser.add_argument('--class_size', type=int, default=99, help='input: class size')
        args = parser.parse_args()

        in_size = ModifiedReferenceCaffeNet.IN_SIZE
        mean = np.load(args.mean_image_path)
        test = DataPreprocessor(args.testing_data_path, args.root_dir_path, mean, in_size, random=False, is_scaled=True)
        iterator = chainer.iterators.SerialIterator(test, batch_size=args.batch_size, repeat=False, shuffle=False)

        # load a model
        model = ModifiedReferenceCaffeNet(args.class_size)
        chainer.serializers.load_npz(args.model_path, model)
        model.select_phase('predict')
        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()  # make the GPU current
            model.to_gpu()

        successful_counts = collections.defaultdict(int)
        total_counts = collections.defaultdict(int)

        # predit
        for batch in iterator:
            xs = []
            ys = []
            for x, y in batch:
                xs.append(x[np.newaxis])
                ys.append(y)
            xss = np.vstack(xs)
            yss = np.vstack(ys)
            if args.gpu >= 0:
                xss = chainer.cuda.to_gpu(xss)

            ts = model(xss, None)
            tss = chainer.cuda.to_cpu(cupy.argmax(ts.data, axis=1))
            yss = yss.reshape(yss.shape[0],)
            bs = tss == yss
            for y, b in zip(yss, bs):
                successful_counts[y] += int(b)
                total_counts[y] += 1

        # calculate accuracies
        for (key, value) in successful_counts.items():
            print('{}: {}'.format(key, value / float(total_counts[key])))

        # calculate total accuracy
        s = sum(successful_counts.values())
        t = sum(total_counts.values())
        print(s / float(t))
    except IOError, e:
        print(e)
