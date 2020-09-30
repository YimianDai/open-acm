from __future__ import division
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn


class DirectAddFuseReduce(HybridBlock):
    def __init__(self, channels=64):
        super(DirectAddFuseReduce, self).__init__()
        self.channels = channels

        self.feature_high = nn.HybridSequential(prefix='feature_high')
        self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                        dilation=1))
        self.feature_high.add(nn.BatchNorm())
        self.feature_high.add((nn.Activation('relu')))

        self.post = nn.HybridSequential(prefix='post')
        self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
        self.post.add(nn.BatchNorm())
        self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)
        xs = xh + xl
        xs = self.post(xs)

        return xs


class DirectAddFuse(HybridBlock):
    def __init__(self, channels=64):
        super(DirectAddFuse, self).__init__()
        self.channels = channels

        self.post = nn.HybridSequential(prefix='post')
        self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
        self.post.add(nn.BatchNorm())
        self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xs = xh + xl
        xs = self.post(xs)

        return xs


class ConcatFuseReduce(HybridBlock):
    def __init__(self, channels=64):
        super(ConcatFuseReduce, self).__init__()
        self.channels = channels

        self.feature_high = nn.HybridSequential(prefix='feature_high')
        self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                        dilation=1))
        self.feature_high.add(nn.BatchNorm())
        self.feature_high.add((nn.Activation('relu')))

        self.post = nn.HybridSequential(prefix='post')
        self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
        self.post.add(nn.BatchNorm())
        self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)
        xs = F.concat(xh, xl, dim=1)
        xs = self.post(xs)

        return xs


class ConcatFuse(HybridBlock):
    def __init__(self, channels=64):
        super(ConcatFuse, self).__init__()
        self.channels = channels

        self.post = nn.HybridSequential(prefix='post')
        self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
        self.post.add(nn.BatchNorm())
        self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xs = F.concat(xh, xl, dim=1)
        xs = self.post(xs)

        return xs


class SKFuseReduce(HybridBlock):
    """Attentional Fusion Strategy developed in Selective Kernel Networks.
    """
    def __init__(self, channels=64, r=4):
        super(SKFuseReduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)
        self.softmax_channels = int(channels * 2)

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_low')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                           dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.GlobalAvgPool2D())
            self.attention.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                         padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('relu'))
            self.attention.add(nn.Conv2D(self.softmax_channels, kernel_size=1, strides=1,
                                         padding=0))
            self.attention.add(nn.BatchNorm())

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)

        xa = xh + xl
        xa = self.attention(xa)  # xa: (B, 2C, 1, 1)
        xa = F.reshape(xa, (0, 2, -1, 0))  # (B, 2, C, 1)
        xa = F.softmax(xa, axis=1)

        xa3 = F.slice_axis(xa, axis=1, begin=0, end=1)  # (B, 1, C, 1)
        xa3 = F.reshape(xa3, (0, -1, 1, 1))
        xa5 = F.slice_axis(xa, axis=1, begin=1, end=2)
        xa5 = F.reshape(xa5, (0, -1, 1, 1))

        xs = 2 * F.broadcast_mul(xh, xa3) + 2 * F.broadcast_mul(xl, xa5)
        xs = self.post(xs)

        return xs


class SKFuse(HybridBlock):
    """Attentional Fusion Strategy developed in Selective Kernel Networks.
    """
    def __init__(self, channels=64, r=4):
        super(SKFuse, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)
        self.softmax_channels = int(channels * 2)

        with self.name_scope():

            self.attention = nn.HybridSequential(prefix='attention')
            self.attention.add(nn.GlobalAvgPool2D())
            self.attention.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                         padding=0))
            self.attention.add(nn.BatchNorm())
            self.attention.add(nn.Activation('relu'))
            self.attention.add(nn.Conv2D(self.softmax_channels, kernel_size=1, strides=1,
                                         padding=0))
            self.attention.add(nn.BatchNorm())

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xa = xh + xl
        xa = self.attention(xa)  # xa: (B, 2C, 1, 1)
        xa = F.reshape(xa, (0, 2, -1, 0))  # (B, 2, C, 1)
        xa = F.softmax(xa, axis=1)

        xa3 = F.slice_axis(xa, axis=1, begin=0, end=1)  # (B, 1, C, 1)
        xa3 = F.reshape(xa3, (0, -1, 1, 1))
        xa5 = F.slice_axis(xa, axis=1, begin=1, end=2)
        xa5 = F.reshape(xa5, (0, -1, 1, 1))

        xs = 2 * F.broadcast_mul(xh, xa3) + 2 * F.broadcast_mul(xl, xa5)
        xs = self.post(xs)

        return xs


class BiLocalChaFuseReduce(HybridBlock):
    """Ablation Structure for Bidirectional Modulation, both use local channel attention
    """
    def __init__(self, channels=64, r=4):
        super(BiLocalChaFuseReduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.topdown = nn.HybridSequential(prefix='topdown')
            self.topdown.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('relu'))
            self.topdown.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('sigmoid'))

            self.bottomup = nn.HybridSequential(prefix='bottomup')
            self.bottomup.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('relu'))
            self.bottomup.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('sigmoid'))

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)
        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * F.broadcast_mul(xl, topdown_wei) + 2 * F.broadcast_mul(xh, bottomup_wei)
        xs = self.post(xs)

        return xs


class BiLocalChaFuse(HybridBlock):
    """Ablation Structure for Bidirectional Modulation, both use local channel attention
    """
    def __init__(self, channels=64, r=4):
        super(BiLocalChaFuse, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)

        with self.name_scope():

            self.topdown = nn.HybridSequential(prefix='topdown')
            self.topdown.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('relu'))
            self.topdown.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('sigmoid'))

            self.bottomup = nn.HybridSequential(prefix='bottomup')
            self.bottomup.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('relu'))
            self.bottomup.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('sigmoid'))

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * F.broadcast_mul(xl, topdown_wei) + 2 * F.broadcast_mul(xh, bottomup_wei)
        xs = self.post(xs)

        return xs


class AsymBiChaFuseReduce(HybridBlock):
    def __init__(self, channels=64, r=4):
        super(AsymBiChaFuseReduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.topdown = nn.HybridSequential(prefix='topdown')
            self.topdown.add(nn.GlobalAvgPool2D())
            self.topdown.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('relu'))
            self.topdown.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('sigmoid'))

            self.bottomup = nn.HybridSequential(prefix='bottomup')
            self.bottomup.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('relu'))
            self.bottomup.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('sigmoid'))

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)
        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * F.broadcast_mul(xl, topdown_wei) + 2 * F.broadcast_mul(xh, bottomup_wei)
        xs = self.post(xs)

        return xs


class AsymBiChaFuse(HybridBlock):
    def __init__(self, channels=64, r=4):
        super(AsymBiChaFuse, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)

        with self.name_scope():

            self.topdown = nn.HybridSequential(prefix='topdown')
            self.topdown.add(nn.GlobalAvgPool2D())
            self.topdown.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('relu'))
            self.topdown.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('sigmoid'))

            self.bottomup = nn.HybridSequential(prefix='bottomup')
            self.bottomup.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('relu'))
            self.bottomup.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('sigmoid'))

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * F.broadcast_mul(xl, topdown_wei) + 2 * F.broadcast_mul(xh, bottomup_wei)
        xs = self.post(xs)

        return xs


class BiGlobalChaFuseReduce(HybridBlock):
    """BiGlobal Modulation for the ablation study
    """
    def __init__(self, channels=64, r=4):
        super(BiGlobalChaFuseReduce, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.topdown = nn.HybridSequential(prefix='topdown')
            self.topdown.add(nn.GlobalAvgPool2D())
            self.topdown.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('relu'))
            self.topdown.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('sigmoid'))

            self.bottomup = nn.HybridSequential(prefix='bottomup')
            self.bottomup.add(nn.GlobalAvgPool2D())
            self.bottomup.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('relu'))
            self.bottomup.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('sigmoid'))

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)
        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * F.broadcast_mul(xl, topdown_wei) + 2 * F.broadcast_mul(xh, bottomup_wei)
        xs = self.post(xs)

        return xs


class BiGlobalChaFuse(HybridBlock):
    """BiGlobal Modulation for the ablation study
    """
    def __init__(self, channels=64, r=4):
        super(BiGlobalChaFuse, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)

        with self.name_scope():

            self.topdown = nn.HybridSequential(prefix='topdown')
            self.topdown.add(nn.GlobalAvgPool2D())
            self.topdown.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('relu'))
            self.topdown.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.topdown.add(nn.BatchNorm())
            self.topdown.add(nn.Activation('sigmoid'))

            self.bottomup = nn.HybridSequential(prefix='bottomup')
            self.bottomup.add(nn.GlobalAvgPool2D())
            self.bottomup.add(nn.Conv2D(self.bottleneck_channels, kernel_size=1, strides=1,
                                          padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('relu'))
            self.bottomup.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                         padding=0))
            self.bottomup.add(nn.BatchNorm())
            self.bottomup.add(nn.Activation('sigmoid'))

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * F.broadcast_mul(xl, topdown_wei) + 2 * F.broadcast_mul(xh, bottomup_wei)
        xs = self.post(xs)

        return xs


class TopDownGlobalChaFuseReduce(HybridBlock):
    """TopDownGlobal Modulation for the ablation study
    """
    def __init__(self, channels=64):
        super(TopDownGlobalChaFuseReduce, self).__init__()
        self.channels = channels

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                          padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)

        xa = xh
        ag = self.global_att(xa)
        xa3 = self.sigmoid(ag)

        xs = xh + F.broadcast_mul(xl, xa3)
        xs = self.post(xs)

        return xs


class TopDownGlobalChaFuse(HybridBlock):
    """TopDownGlobal Modulation for the ablation study
    """
    def __init__(self, channels=64):
        super(TopDownGlobalChaFuse, self).__init__()
        self.channels = channels

        with self.name_scope():

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.GlobalAvgPool2D())
            self.global_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                          padding=0))
            self.global_att.add(nn.BatchNorm())

            self.sigmoid = nn.Activation('sigmoid')

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xa = xh
        ag = self.global_att(xa)
        xa3 = self.sigmoid(ag)

        xs = xh + F.broadcast_mul(xl, xa3)
        xs = self.post(xs)

        return xs


class TopDownLocalChaFuseReduce(HybridBlock):
    """TopDownLocal Modulation for the ablation study
    """
    def __init__(self, channels=64):
        super(TopDownLocalChaFuseReduce, self).__init__()
        self.channels = channels

        with self.name_scope():

            self.feature_high = nn.HybridSequential(prefix='feature_high')
            self.feature_high.add(nn.Conv2D(channels, kernel_size=1, strides=1, padding=0,
                                            dilation=1))
            self.feature_high.add(nn.BatchNorm())
            self.feature_high.add(nn.Activation('relu'))

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                          padding=0))
            self.global_att.add(nn.BatchNorm())
            self.sigmoid = nn.Activation('sigmoid')

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xh = self.feature_high(xh)

        xa = xh
        ag = self.global_att(xa)
        xa3 = self.sigmoid(ag)

        xs = xh + F.broadcast_mul(xl, xa3)
        xs = self.post(xs)

        return xs


class TopDownLocalChaFuse(HybridBlock):
    """TopDownLocal Modulation for the ablation study
    """
    def __init__(self, channels=64):
        super(TopDownLocalChaFuse, self).__init__()
        self.channels = channels

        with self.name_scope():

            self.global_att = nn.HybridSequential(prefix='global_att')
            self.global_att.add(nn.Conv2D(self.channels, kernel_size=1, strides=1,
                                          padding=0))
            self.global_att.add(nn.BatchNorm())
            self.sigmoid = nn.Activation('sigmoid')

            self.post = nn.HybridSequential(prefix='post')
            self.post.add(nn.Conv2D(channels, kernel_size=3, strides=1, padding=1, dilation=1))
            self.post.add(nn.BatchNorm())
            self.post.add((nn.Activation('relu')))

    def hybrid_forward(self, F, xh, xl):

        xa = xh
        ag = self.global_att(xa)
        xa3 = self.sigmoid(ag)

        xs = xh + F.broadcast_mul(xl, xa3)
        xs = self.post(xs)

        return xs




