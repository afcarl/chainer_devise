#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
import chainer.cuda


class DeviseInFirstStage(chainer.Chain):
    MARGIN = 0.1

    def __init__(self, visual_feature_size, word2vec_size):
        super(DeviseInFirstStage, self).__init__(
            fc=L.Linear(visual_feature_size, word2vec_size)
        )

    # test ok
    @staticmethod
    def calculate_batch_matmul(u, tl, tk):
        '''
            bs: batch_size
            ws: word2vec_size
            ss: sample_size
            u.shape:  (bs, ws)
            tl.shape: (bs, ws, 1)
            tk.shape: (bs, ws, ss)
        '''
        bs, ws, ss = tk.shape
        tl = F.broadcast_to(tl, (bs, ws, ss))
        tm = tl - tk  # (bs, ws, ss)
        return F.batch_matmul(tm, u, transa=True)  # (bs, ss, 1)

    # test ok
    @staticmethod
    def calculate_loss(u, tl, tk):
        '''
            u.shape = (bs, ws)
            tl.shape = (bs, ws, 1)
            tk.shape = (bs, ws, ss)
        '''
        xp = chainer.cuda.get_array_module(u)
        c = DeviseInFirstStage.calculate_batch_matmul(u, tl, tk)  # (bs, ss, 1)
        d = DeviseInFirstStage.MARGIN - c  # (bs, ss, 1)
        e = F.maximum(chainer.Variable(xp.zeros(d.shape, xp.float32)), d)  # (bs, ss, 1)
        return F.sum(e, axis=1)  # (bs, 1)

    # test ok
    def __call__(self, v, t):
        '''
            bs: batch_size
            vs: visual_feature_size
            ws: word2vec_size
            ss: sample_size
            v.shape: (bs, vs)
            t.shape: (bs, ws, 1 + ss)
        '''

        bs, _, vs = v.shape
        v = F.reshape(v, (bs, vs))
        tl, tk = F.split_axis(t, [1], axis=2)
        u = self.fc(v)  # (bs, ws)
        loss = DeviseInFirstStage.calculate_loss(u, tl, tk)  # (bs, 1)
        chainer.report({'loss': loss}, self)
        print(loss.data)
        return loss


if __name__ == '__main__':
    pass
