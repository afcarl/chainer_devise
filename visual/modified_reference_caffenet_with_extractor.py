import chainer
import chainer.functions as F
import chainer.links as L


class ModifiedReferenceCaffeNetWithExtractor(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    IN_SIZE = 227

    def __init__(self, class_size):
        super(ModifiedReferenceCaffeNetWithExtractor, self).__init__(
            conv1=L.Convolution2D(3, 96, 11, stride=4),  # pad=0
            conv2=L.Convolution2D(96, 256,  5, pad=2),  # stride=1
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(9216, 4096),  # 9216=6x6x256
            fc7=L.Linear(4096, 4096),
            modified_fc8=L.Linear(4096, class_size),
        )
        self.select_phase('train')

    def select_phase(self, phase):
        if phase == 'predict':
            self.extractor = False
            self.train = False
            self.predict = True
        elif phase == 'train':
            self.extractor = False
            self.train = True
            self.predict = False
        elif phase == 'test':
            self.extractor = False
            self.train = False
            self.predict = False
        elif phase == 'extractor':
            self.train = False
            self.predict = False
            self.extractor = True
        else:
            raise Exception('unknown phase')

    def __call__(self, x, t):

        # conv1->relu1->pool1->norm1
        h = F.local_response_normalization(
            F.max_pooling_2d(
                F.relu(
                    self.conv1(x)
                ),
                ksize=3,
                stride=2
                # pad=0
            )
            # n(local_size)=5
            # alpha=0.0001
            # beta=0.75
        )

        # conv2->relu2->pool2->norm2
        h = F.local_response_normalization(
            F.max_pooling_2d(
                F.relu(
                    self.conv2(h)
                ),
                ksize=3,
                stride=2
            )
        )

        # conv3->relu3
        h = F.relu(self.conv3(h))

        # conv4->relu4
        h = F.relu(self.conv4(h))

        # conv5->relu5->pooling5
        h = F.max_pooling_2d(
            F.relu(
                self.conv5(h)
            ),
            ksize=3,
            stride=2
        )

        # fc6->relu6->drop6
        h = F.dropout(
            F.relu(
                self.fc6(h)
            ),
            train=self.train,
            ratio=0.5
        )

        # fc7->relu7->drop7
        h = F.dropout(
            F.relu(
                self.fc7(h)
            ),
            train=self.train,
            ratio=0.5
        )

        # for extractor
        value_for_extractor = h

        # modified fc8
        h = self.modified_fc8(h)

        if self.predict:
            return F.softmax(h)
        elif self.extractor:
            # REMOVE THE SECOND TERM IN THE FUTURE!!!
            return value_for_extractor, F.softmax(h)
        else:
            loss = F.softmax_cross_entropy(h, t)
            accuracy = F.accuracy(h, t)
            chainer.report({'loss': loss, 'accuracy': accuracy}, self)
            return loss
