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
            tl.shape: (bs, ws, 1)
            tk.shape: (bs, ws, ss)
        '''
        bs, ws, ss = tk.shape
        tl = F.broadcast_to(tl, (bs, ws, ss))
        tm = tl - tk  # (bs, ws, ss)
        return F.batch_matmul(tm, u)  # (bs, ss)

    # test ok
    def calculate_loss(self, u, tl, tk):
        '''
            u.shape = (bs, vs)
            tl.shape = (bs, vs, 1)
            tk.shape = (bs, vs, ss)
        '''
        xp = chainer.cuda.get_array_module(u)
        c = self.calculate_batch_matmul(u, tl, tk)  # (bs, ss)
        d = DeviseInFirstStage.MARGIN - c  # (bs, ss)
        e = F.maximum(chainer.Variable(xp.zeros(d.shape, xp.float32)), d)
        return F.sum(e, axis=1)  # (bs,)

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

    def __call__(self, v, t):
        '''
            bs: batch_size
            vs: visual_feature_size
            ws: word2vec_size
            ss: sample_size
            v.shape: (bs, vs)
            tl.shape: (bs, ws, 1 + ss)
        '''

        bs, _, vs = v.shape
        v = F.reshape(v, (bs, vs))
        tl, tk = F.split_axis(t, [1], axis=2)
        print('type(v):{}, v.shape:{}'.format(type(v), v.shape))
        print('type(tl):{}, tl.shape:{}'.format(type(tl), tl.shape))
        print('type(tk):{}, tk.shape:{}'.format(type(tk), tk.shape))
        u = self.fc(v)
        loss = self.calculate_loss(u, tl, tk)
        chainer.report({'loss': loss}, self)
        return loss


if __name__ == '__main__':
    pass
