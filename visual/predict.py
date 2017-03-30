#!/usr/bin/env python
# coding:utf-8

from modified_reference_caffenet import *  # noqa
import chainer
import numpy as np


PATH = '/home/ubuntu/results/devise/model_iter_10'

if __name__ == '__main__':
    class_size = 207
    model = ModifiedReferenceCaffeNet(class_size)
    chainer.serializers.load_npz(PATH, model)
    print(model)
    model.select_phase('predict')
    x = np.ones((1, 3, 227, 227)).astype(np.float32)
    y = model(x, None)
