#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from modified_reference_caffenet import *
from copy_model import *
import cPickle

PICKLE_PATH = "/home/ubuntu/data/models/chainer/bvlc_reference_caffenet/bvlc_reference_caffenet-2017-01-08.pkl"
#PICKLE_PATH = "/home/ubuntu/data/models/chainer/bvlc_reference_caffenet/bvlc_reference_caffenet.pkl"

if __name__ == "__main__":

    print("load caffe model")
    caffe_model = cPickle.load(open(PICKLE_PATH))
    
    print("copy weights")
    model = ModifiedReferenceCaffeNet()
    copy_model(caffe_model, model)

    pass

