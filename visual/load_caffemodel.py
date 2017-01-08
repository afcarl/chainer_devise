#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer.links.caffe
import cPickle

CAFFE_MODEL_PATH = "/home/ubuntu/data/models/caffe/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
PICKLE_PATH = "/home/ubuntu/data/models/chainer/bvlc_reference_caffenet/bvlc_reference_caffenet-2017-01-08.pkl"

if __name__ == "__main__":
    model = chainer.links.caffe.caffe_function.CaffeFunction(CAFFE_MODEL_PATH)
    cPickle.dump(model, open(PICKLE_PATH, "wb"))
