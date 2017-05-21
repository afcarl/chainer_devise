#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import chainer
import chainer.links as L
import chainer.functions as F


class DeviseInFirstStage(chainer.Chain):
    MARGIN = 0.1

    def __init__(self, visual_feature_size, word2vec_size):
        super(DeviseInFirstStage, self).__init__(
            fc=L.Linear(visual_feature_size, word2vec_size)
        )

    def calculate_loss(self, x, cy, ny):
        # x.shape, cy.shape, cx.shape: (batch_size, word2vec_size)
        h = DeviseInFirstStage.MARGIN - x.dot(cy) + x.dot(ny)
        return F.max(h)

    def __call__(self, vx, cwx, nwx):
        # vx.shape: (batch_size, visual_feature_size)
        # cwx.shape: (batch_size, word2vec_size)
        # nwx.shape: (sample_size, word2vec_size)
        h = self.fc(vx)  # (batch_size, word2vec_size)
        loss = self.calculate_loss(h, cwx, nwx)
        chainer.report({'loss': loss}, self)
        return loss


if __name__ == '__main__':
    pass
