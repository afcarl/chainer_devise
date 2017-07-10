#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F
import chainer.cuda


class DeviseInFirstStage(chainer.Chain):
    MARGIN = 0.1
    IN_SIZE = 227

    def __init__(self, visual_feature_size, word2vec_size):
        super(DeviseInFirstStage, self).__init__(
            fc=L.Linear(visual_feature_size, word2vec_size)
        )

    # test ok
    def calculate_batch_matmul(self, u, tl, tk):
        '''
            bs: batch_size
            ws: word2vec_size
            ss: sample_size
            u.shape:  (bs, ws)
            tl.shape: (bs, ws)
            tk.shape: (bs, ss, ws)
        '''
        bs, ss, ws = tk.shape
        tl = F.broadcast_to(tl[:, None, :], (bs, ss, ws))
        tm = tl - tk  # (bs, ss, ws)
        c = F.batch_matmul(tm, u)
        return F.reshape(c, (bs, ss))

    # test ok
    def calculate_loss(self, u, tl, tk):
        xp = chainer.cuda.get_array_module(u)
        c = self.calculate_batch_matmul(u, tl, tk)
        d = DeviseInFirstStage.MARGIN - c
        e = F.maximum(chainer.Variable(xp.zeros(d.shape, xp.float32)), d)
        return F.sum(e, axis=1)

    # test ok
    # def __call__(self, v, tl, tk):
    #     '''
    #         bs: batch_size
    #         vs: visual_feature_size
    #         ws: word2vec_size
    #         ss: sample_size
    #         v.shape: (bs, vs)
    #         tl.shape: (bs, ws)
    #         tk.shape: (bs, ws, ss)
    #     '''
    #     u = self.fc(v)  # (bs, ws)
    #     loss = self.calculate_loss(u, tl, tk)
    #     chainer.report({'loss': loss}, self)
    #     return loss

    def __call__(self, v, tl):
        bs, _, vs = v.shape
        v = F.reshape(v, (bs, vs))
        print('type(v):{}, v.shape:{}'.format(type(v), v.shape))
        print('type(tl):{}, tl.shape:{}'.format(type(tl), tl.shape))
        return 0.1


if __name__ == '__main__':
    pass
