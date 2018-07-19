from mxnet.gluon import nn, HybridBlock

class BottleneckV2(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm(epsilon=2e-5, use_global_stats=True)
        self.conv1 = nn.Conv2D(channels // 4, kernel_size=1, strides=1, use_bias=False)
        self.bn2 = nn.BatchNorm(epsilon=2e-5, use_global_stats=True)
        self.conv2 = nn.Conv2D(channels // 4, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=channels // 4)
        self.bn3 = nn.BatchNorm(epsilon=2e-5, use_global_stats=True)
        self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False)
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv3(x)

        return x + residual


class ResNetV2a(HybridBlock):
    def __init__(self, layers, **kwargs):
        super(ResNetV2a, self).__init__(**kwargs)
        with self.name_scope():
            self.layer0 = nn.HybridSequential(prefix='')
            self.layer0.add(nn.BatchNorm(scale=False, epsilon=2e-5, use_global_stats=True))
            self.layer0.add(nn.Conv2D(64, 7, 2, 3, use_bias=False))
            self.layer0.add(nn.BatchNorm(epsilon=2e-5, use_global_stats=True))
            self.layer0.add(nn.Activation('relu'))
            self.layer0.add(nn.MaxPool2D(3, 2, 1))

            self.layer1 = self._make_layer(stage_index=1, layers=layers[0], in_channels=64, channels=256, stride=1)
            self.layer2 = self._make_layer(stage_index=2, layers=layers[1], in_channels=256, channels=512, stride=2)
            self.layer3 = self._make_layer(stage_index=3, layers=layers[2], in_channels=512, channels=1024, stride=2)
            self.layer4 = self._make_layer(stage_index=4, layers=layers[3], in_channels=1024, channels=2048, stride=2)

            self.layer4.add(nn.BatchNorm(epsilon=2e-5, use_global_stats=True))
            self.layer4.add(nn.Activation('relu'))
            self.layer4.add(nn.GlobalAvgPool2D())
            self.layer4.add(nn.Flatten())

    def _make_layer(self, stage_index, layers, channels, stride, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(BottleneckV2(channels, stride, channels != in_channels, in_channels=in_channels, prefix=''))
            for _ in range(layers - 1):
                layer.add(BottleneckV2(channels, 1, False, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
