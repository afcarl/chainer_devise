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

    def test_call(self):
        bs = 1
        vs = 5

        v = np.arange(bs * vs, dtype=np.float32)
        v = v.reshape(bs, 1, vs)
        v = chainer.Variable(v)

        ws = 3
        devise = DeviseInFirstStage(vs, ws)
        a = np.zeros((ws, vs))
        a[1, 1] = 1
        a[2, 2] = 1
        devise.fc.W.data = a

        tl = np.arange(bs * ws, dtype=np.float32)
        tl = tl.reshape(bs, ws, 1)

        ss = 2
        tk = np.arange(bs * ws * ss, dtype=np.float32)
        tk = tk.reshape(bs, ws, ss)

        t = np.concatenate((tl, tk), axis=2)
        t = chainer.Variable(t)

        loss = devise(v, t)
        self.assertAlmostEqual(loss.data[0, 0], 13.20000076, delta=1.0e-05)


if __name__ == '__main__':
    unittest.main()
