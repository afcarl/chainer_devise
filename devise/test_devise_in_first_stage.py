#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from devise_in_first_stage import *  # noqa
import unittest
import chainer
import numpy as np
import chainer.functions as F


class TestDeviseInFirstStage(unittest.TestCase):

    def test_broadcast(self):
        bs = 3
        ws = 2
        a = np.arange(bs * ws)
        a = a.reshape(bs, ws, 1)

        v = chainer.Variable(a)
        ss = 4
        u = F.broadcast_to(v, (bs, ws, ss))
        for i in range(ss):
            self.assertTrue(np.all(u.data[:, :, i] == a[:, :, 0]))

    def test_batch_matmul(self):
        bs = 2
        ws = 3
        ss = 4
        tm = np.arange(bs * ws * ss, dtype=np.float32)
        tm = chainer.Variable(tm.reshape(bs, ws, ss))  # bs (ws,ss)
        u = np.arange(bs * ws, dtype=np.float32)
        u = chainer.Variable(u.reshape(bs, ws))  # bs (ws,1)
        y = F.batch_matmul(tm, u, transa=True)  # bs (ss,ws) * (ws,1) -> bs (ss,1)
        self.assertTrue(y.data.shape == (bs, ss, 1))

    def test_calculate_batch_matmul(self):
        bs = 1
        ws = 3
        ss = 2

        tl = np.arange(bs * ws, dtype=np.float32)
        tl = chainer.Variable(tl.reshape(bs, ws, 1))
        # print('tl', tl.data)

        tk = np.arange(bs * ws * ss, dtype=np.float32)
        tk = chainer.Variable(tk.reshape(bs, ws, ss))
        # print('tk', tk.data)

        u = np.arange(bs * ws, dtype=np.float32)
        u = chainer.Variable(u.reshape(bs, ws))
        # print('u', u.data)

        y = DeviseInFirstStage.calculate_batch_matmul(u, tl, tk)
        self.assertTrue(y.data.shape == (bs, ss, 1))
        self.assertTrue(np.all(y.data == np.array([[[-5], [-8]]])))

    def test_calculate_loss(self):
        bs = 1
        ws = 3
        ss = 2

        tl = np.arange(bs * ws, dtype=np.float32)
        tl = chainer.Variable(tl.reshape(bs, ws, 1))
        # print('tl', tl.data)

        tk = np.arange(bs * ws * ss, dtype=np.float32)
        tk = chainer.Variable(tk.reshape(bs, ws, ss))
        # print('tk', tk.data)

        u = np.arange(bs * ws, dtype=np.float32)
        u = chainer.Variable(u.reshape(bs, ws))
        # print('u', u.data)

        y = DeviseInFirstStage.calculate_loss(u, tl, tk)
        self.assertAlmostEqual(y.data[0, 0], 13.20000076, delta=1.0e-05)

    def test_linear(self):
        bs = 2
        vs = 5

        v = np.arange(bs * vs, dtype=np.float32)
        v = v.reshape(bs, vs)

        vv = chainer.Variable(v)

        ws = 3
        devise = DeviseInFirstStage(vs, ws)
        r = devise.fc(vv)
        self.assertTrue(r.data.shape == (bs, ws))
        print(r.data)

    # def test_call(self):
    #     bs = 2
    #     ss = 3
    #     ws = 4
    #     vs = 5

    #     # visual vector
    #     v = np.array([[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]], dtype=np.float32)
    #     v = v.reshape(bs, 1, vs)
    #     vv = chainer.Variable(v)
    #     '''
    #         0 1 2 3 4
    #         5 6 7 8 9
    #     '''

    #     # correct label
    #     tl = np.arange(bs * ws, dtype=np.float32)
    #     tl = tl.reshape(bs, ws, 1)
    #     # vtl = chainer.Variable(tl)
    #     '''
    #         0 1 2 3
    #         4 5 6 7
    #     '''

    #     # negative label
    #     tk = np.arange(bs * ss * ws, dtype=np.float32)
    #     tk = tk.reshape(bs, ws, ss)
    #     # vtk = chainer.Variable(tk)
    #     '''
    #         0 1  2  3
    #         4 5  6  7
    #         8 9 10 11

    #         12 13 14 15
    #         16 17 18 19
    #         20 21 22 23
    #     '''
    #     t = np.concatenate((tl, tk), axis=2)
    #     vt = chainer.Variable(t)
    #     self.assertTrue(t.shape == (bs, ws, 1 + ss))
    #     devise = DeviseInFirstStage(vs, ws)
    #     devise.fc.W.data = np.eye(ws, vs)
    #     c = devise(vv, vt)
    #     # self.assertAlmostEqual(c.data[0], 72.30000305, delta=1.0e-05)
    #     # self.assertAlmostEqual(c.data[1], 792.30004883, delta=1.0e-05)


if __name__ == '__main__':
    unittest.main()
