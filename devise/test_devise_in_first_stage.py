#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from devise_in_first_stage import *  # noqa
import unittest
import chainer
import numpy as np
import chainer.functions as F


class TestDeviseInFirstStage(unittest.TestCase):

    def test_broadcast(self):
        row = 3
        col = 2
        a = np.arange(row * col)
        a = a.reshape(row, col)

        v = chainer.Variable(a)
        u = F.broadcast_to(v[:, None, :], (row, 4, col))
        self.assertTrue(np.all(u.data[:, 0, :] == a))
        self.assertTrue(np.all(u.data[:, 1, :] == a))
        self.assertTrue(np.all(u.data[:, 2, :] == a))
        self.assertTrue(np.all(u.data[:, 3, :] == a))

    def test_batch_matmul(self):
        bs = 2
        n = 3
        m = 4

        a = np.arange(bs * n, dtype=np.float32)
        a = a.reshape(bs, n)

        b = np.arange(bs * m * n, dtype=np.float32)
        b = b.reshape(bs, m, n)

        '''
        b[0] = 0  1  2
               3  4  5
               6  7  8
               9 10 11

        a[0] = 0
               1
               2

        b[0] a[0] =  5
                    14
                    23
                    32

        b[1] = 12 13 14
               15 16 17
               18 19 20
               21 22 23

        a[1] = 3
               4
               5

        b[1] a[1] = 158
                    194
                    230
                    266
        '''

        a = chainer.Variable(a)
        b = chainer.Variable(b)
        c = F.batch_matmul(b, a)
        self.assertTrue((bs, m, 1) == c.data.shape)
        c = F.reshape(c, (bs, m))
        self.assertTrue(np.all([5, 14, 23, 32] == c.data[0]))
        self.assertTrue(np.all([158, 194, 230, 266] == c.data[1]))

    def test_calculate_batch_matmul(self):
        bs = 2
        ss = 3
        ws = 4

        u = np.arange(bs * ws, dtype=np.float32)
        u = u.reshape(bs, ws)
        '''
            0 1 2 3
            4 5 6 7
        '''

        vu = chainer.Variable(u)

        tl = np.arange(bs * ws, dtype=np.float32)
        tl = tl.reshape(bs, ws)
        '''
            0 1 2 3
            4 5 6 7
        '''

        vtl = chainer.Variable(tl)
        '''
        hoge = F.broadcast_to(tl[:, None, :], (bs, ss, ws))
            0 1 2 3
            0 1 2 3
            0 1 2 3

            4 5 6 7
            4 5 6 7
            4 5 6 7
        '''
        tk = np.arange(bs * ss * ws, dtype=np.float32)
        tk = tk.reshape(bs, ss, ws)
        '''
            0 1  2  3
            4 5  6  7
            8 9 10 11

            12 13 14 15
            16 17 18 19
            20 21 22 23
        '''

        '''
        tm = hoge - tk
             0  0  0  0
            -4 -4 -4 -4
            -8 -8 -8 -8

             -8  -8  -8  -8
            -12 -12 -12 -12
            -16 -16 -16 -16
        '''
        vtk = chainer.Variable(tk)

        devise = DeviseInFirstStage(1, 1)
        c = devise.calculate_batch_matmul(vu, vtl, vtk)
        self.assertTrue((bs, ss) == c.data.shape)
        self.assertTrue(np.all(c.data == [[0, -24, -48], [-176, -264, -352]]))

    def test_calculate_loss(self):
        bs = 2
        ss = 3
        ws = 4

        u = np.arange(bs * ws, dtype=np.float32)
        u = u.reshape(bs, ws)
        '''
            0 1 2 3
            4 5 6 7
        '''

        vu = chainer.Variable(u)

        tl = np.arange(bs * ws, dtype=np.float32)
        tl = tl.reshape(bs, ws)
        '''
            0 1 2 3
            4 5 6 7
        '''

        vtl = chainer.Variable(tl)
        '''
        hoge = F.broadcast_to(tl[:, None, :], (bs, ss, ws))
            0 1 2 3
            0 1 2 3
            0 1 2 3

            4 5 6 7
            4 5 6 7
            4 5 6 7
        '''
        tk = np.arange(bs * ss * ws, dtype=np.float32)
        tk = tk.reshape(bs, ss, ws)
        '''
            0 1  2  3
            4 5  6  7
            8 9 10 11

            12 13 14 15
            16 17 18 19
            20 21 22 23
        '''

        '''
        tm = hoge - tk
             0  0  0  0
            -4 -4 -4 -4
            -8 -8 -8 -8

             -8  -8  -8  -8
            -12 -12 -12 -12
            -16 -16 -16 -16
        '''
        vtk = chainer.Variable(tk)

        devise = DeviseInFirstStage(1, 1)
        c = devise.calculate_loss(vu, vtl, vtk)
        self.assertAlmostEqual(c.data[0], 72.30000305, delta=1.0e-05)
        self.assertAlmostEqual(c.data[1], 792.30004883, delta=1.0e-05)


if __name__ == '__main__':
    unittest.main()
