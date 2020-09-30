from __future__ import division

from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from gluoncv.model_zoo.fcn import _FCNHead
from gluoncv.model_zoo.cifarresnet import CIFARBasicBlockV1

from .fusion import DirectAddFuseReduce, ConcatFuseReduce, SKFuseReduce, BiLocalChaFuseReduce, \
    BiGlobalChaFuseReduce, AsymBiChaFuseReduce, TopDownGlobalChaFuseReduce, \
    TopDownLocalChaFuseReduce
from .fusion import DirectAddFuse, ConcatFuse, SKFuse, BiLocalChaFuse, BiGlobalChaFuse, \
    AsymBiChaFuse, TopDownGlobalChaFuse, TopDownLocalChaFuse


class ASKCResNetFPN(HybridBlock):
    def __init__(self, layers, channels, fuse_mode, tiny=True, classes=1,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(ASKCResNetFPN, self).__init__(**kwargs)

        self.layer_num = len(layers)
        self.tiny = tiny
        with self.name_scope():

            stem_width = int(channels[0])
            self.stem = nn.HybridSequential(prefix='stem')
            self.stem.add(norm_layer(scale=False, center=False,
                                     **({} if norm_kwargs is None else norm_kwargs)))
            if tiny:
                self.stem.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.stem.add(norm_layer(in_channels=stem_width*2))
                self.stem.add(nn.Activation('relu'))
            else:
                # self.stem.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=2,
                #                          padding=1, use_bias=False))
                # self.stem.add(norm_layer(in_channels=stem_width*2))
                # self.stem.add(nn.Activation('relu'))
                # self.stem.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
                self.stem.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                         padding=1, use_bias=False))
                self.stem.add(norm_layer(in_channels=stem_width))
                self.stem.add(nn.Activation('relu'))
                self.stem.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.stem.add(norm_layer(in_channels=stem_width))
                self.stem.add(nn.Activation('relu'))
                self.stem.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.stem.add(norm_layer(in_channels=stem_width*2))
                self.stem.add(nn.Activation('relu'))
                self.stem.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))

            self.head = _FCNHead(in_channels=channels[1], channels=classes)

            self.layer1 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[0],
                                           channels=channels[1], stride=1, stage_index=1,
                                           in_channels=channels[1])

            self.layer2 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[1],
                                           channels=channels[2], stride=2, stage_index=2,
                                           in_channels=channels[1])

            self.layer3 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[2],
                                           channels=channels[3], stride=2, stage_index=3,
                                           in_channels=channels[2])

            if self.layer_num == 4:
                self.layer4 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[3],
                                               channels=channels[4], stride=2, stage_index=4,
                                               in_channels=channels[3])
                self.fuse34 = self._fuse_layer(fuse_mode, channels=channels[3])  # channels[4]

            self.fuse23 = self._fuse_layer(fuse_mode, channels=channels[2])  # 64
            self.fuse12 = self._fuse_layer(fuse_mode, channels=channels[1])  # 32

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            downsample = (channels != in_channels) or (stride != 1)
            layer.add(block(channels, stride, downsample, in_channels=in_channels,
                            prefix='', norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix='',
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def _fuse_layer(self, fuse_mode, channels):
        if fuse_mode == 'DirectAdd':
            fuse_layer = DirectAddFuseReduce(channels=channels)
        elif fuse_mode == 'Concat':
            fuse_layer = ConcatFuseReduce(channels=channels)
        elif fuse_mode == 'SK':
            fuse_layer = SKFuseReduce(channels=channels)
        elif fuse_mode == 'BiLocal':
            fuse_layer = BiLocalChaFuseReduce(channels=channels)
        elif fuse_mode == 'BiGlobal':
            fuse_layer = BiGlobalChaFuseReduce(channels=channels)
        elif fuse_mode == 'AsymBi':
            fuse_layer = AsymBiChaFuseReduce(channels=channels)
        elif fuse_mode == 'TopDownGlobal':
            fuse_layer = TopDownGlobalChaFuseReduce(channels=channels)
        elif fuse_mode == 'TopDownLocal':
            fuse_layer = TopDownLocalChaFuseReduce(channels=channels)
        else:
            raise ValueError('Unknown fuse_mode')

        return fuse_layer

    def hybrid_forward(self, F, x):

        _, _, hei, wid = x.shape

        x = self.stem(x)      # down 4, 32
        c1 = self.layer1(x)   # down 4, 32
        c2 = self.layer2(c1)  # down 8, 64
        out = self.layer3(c2)  # down 16, 128
        if self.layer_num == 4:
            c4 = self.layer4(out)  # down 32
            if self.tiny:
                c4 = F.contrib.BilinearResize2D(c4, height=hei//4, width=wid//4)  # down 4
            else:
                c4 = F.contrib.BilinearResize2D(c4, height=hei//16, width=wid//16)  # down 16
            out = self.fuse34(c4, out)
        if self.tiny:
            out = F.contrib.BilinearResize2D(out, height=hei//2, width=wid//2)  # down 2, 128
        else:
            out = F.contrib.BilinearResize2D(out, height=hei//8, width=wid//8)  # down 8, 128
        out = self.fuse23(out, c2)
        if self.tiny:
            out = F.contrib.BilinearResize2D(out, height=hei, width=wid)  # down 1
        else:
            out = F.contrib.BilinearResize2D(out, height=hei//4, width=wid//4)  # down 8
        out = self.fuse12(out, c1)

        pred = self.head(out)
        if self.tiny:
            out = pred
        else:
            out = F.contrib.BilinearResize2D(pred, height=hei, width=wid)  # down 4

        return out

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)


class ASKCResUNet(HybridBlock):
    def __init__(self, layers, channels, fuse_mode, tiny=True, classes=1,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(ASKCResUNet, self).__init__(**kwargs)

        self.layer_num = len(layers)
        self.tiny = tiny
        with self.name_scope():

            stem_width = int(channels[0])
            self.stem = nn.HybridSequential(prefix='stem')
            self.stem.add(norm_layer(scale=False, center=False,
                                     **({} if norm_kwargs is None else norm_kwargs)))
            if tiny:
                self.stem.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.stem.add(norm_layer(in_channels=stem_width*2))
                self.stem.add(nn.Activation('relu'))
            else:
                # self.stem.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=2,
                #                          padding=1, use_bias=False))
                # self.stem.add(norm_layer(in_channels=stem_width*2))
                # self.stem.add(nn.Activation('relu'))
                # self.stem.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))

                self.stem.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                         padding=1, use_bias=False))
                self.stem.add(norm_layer(in_channels=stem_width))
                self.stem.add(nn.Activation('relu'))
                self.stem.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.stem.add(norm_layer(in_channels=stem_width))
                self.stem.add(nn.Activation('relu'))
                self.stem.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.stem.add(norm_layer(in_channels=stem_width*2))
                self.stem.add(nn.Activation('relu'))
                self.stem.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))

            self.layer1 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[0],
                                           channels=channels[1], stride=1, stage_index=1,
                                           in_channels=channels[1])

            self.layer2 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[1],
                                           channels=channels[2], stride=2, stage_index=2,
                                           in_channels=channels[1])

            self.layer3 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[2],
                                           channels=channels[3], stride=2, stage_index=3,
                                           in_channels=channels[2])

            self.deconv2 = nn.Conv2DTranspose(channels=channels[2], kernel_size=(4, 4),
                                              strides=(2, 2), padding=1)
            self.uplayer2 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[1],
                                             channels=channels[2], stride=1, stage_index=4,
                                             in_channels=channels[2])
            self.fuse2 = self._fuse_layer(fuse_mode, channels=channels[2])

            self.deconv1 = nn.Conv2DTranspose(channels=channels[1], kernel_size=(4, 4),
                                              strides=(2, 2), padding=1)
            self.uplayer1 = self._make_layer(block=CIFARBasicBlockV1, layers=layers[0],
                                             channels=channels[1], stride=1, stage_index=5,
                                             in_channels=channels[1])
            self.fuse1 = self._fuse_layer(fuse_mode, channels=channels[1])

            self.head = _FCNHead(in_channels=channels[1], channels=classes)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    norm_layer=BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            downsample = (channels != in_channels) or (stride != 1)
            layer.add(block(channels, stride, downsample, in_channels=in_channels,
                            prefix='', norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix='',
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        return layer

    def _fuse_layer(self, fuse_mode, channels):
        if fuse_mode == 'DirectAdd':
            fuse_layer = DirectAddFuse(channels=channels)
        elif fuse_mode == 'Concat':
            fuse_layer = ConcatFuse(channels=channels)
        elif fuse_mode == 'SK':
            fuse_layer = SKFuse(channels=channels)
        elif fuse_mode == 'BiLocal':
            fuse_layer = BiLocalChaFuse(channels=channels)
        elif fuse_mode == 'BiGlobal':
            fuse_layer = BiGlobalChaFuse(channels=channels)
        elif fuse_mode == 'AsymBi':
            fuse_layer = AsymBiChaFuse(channels=channels)
        elif fuse_mode == 'TopDownGlobal':
            fuse_layer = TopDownGlobalChaFuse(channels=channels)
        elif fuse_mode == 'TopDownLocal':
            fuse_layer = TopDownLocalChaFuse(channels=channels)
        else:
            raise ValueError('Unknown fuse_mode')

        return fuse_layer

    def hybrid_forward(self, F, x):

        _, _, hei, wid = x.shape

        x = self.stem(x)      # 480x480, 16
        c1 = self.layer1(x)   # 480x480, 16
        c2 = self.layer2(c1)  # 240x240, 32
        c3 = self.layer3(c2)  # 120x120, 64

        deconvc2 = self.deconv2(c3)  # 240x240, 32
        fusec2 = self.fuse2(deconvc2, c2)  # 240x240, 32
        upc2 = self.uplayer2(fusec2)  # 240x240, 32

        deconvc1 = self.deconv1(upc2)  # 480x480, 16
        fusec1 = self.fuse1(deconvc1, c1)  # 480x480, 16
        upc1 = self.uplayer1(fusec1)  # 240x240, 32

        pred = self.head(upc1)

        if self.tiny:
            out = pred
        else:
            out = F.contrib.BilinearResize2D(pred, height=hei, width=wid)  # down 4

        return out

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)
