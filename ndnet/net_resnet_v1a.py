from mxnet.gluon import nn, HybridBlock

class BottleneckV1(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(channels // 4, kernel_size=1, strides=stride, use_bias=False)
        self.bn1 = nn.BatchNorm(use_global_stats=True)
        self.conv2 = nn.Conv2D(channels // 4, kernel_size=3, strides=1, padding=1, use_bias=False)
        self.bn2 = nn.BatchNorm(use_global_stats=True)
        self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False)
        self.bn3 = nn.BatchNorm(use_global_stats=True)
        if downsample:
            self.downsample = nn.HybridSequential(prefix='down_')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride, use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm(use_global_stats=True))
        else:
            self.downsample = None
        self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample:
            residual = self.downsample(residual)
        x = F.Activation(x + residual, act_type='relu')
        return x


class ResNetV1a(HybridBlock):
    def __init__(self, layers, **kwargs):
        super(ResNetV1a, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(64, kernel_size=7, strides=2, padding=3, use_bias=False)
            self.bn1 = nn.BatchNorm(use_global_stats=True)
            self.relu = nn.Activation('relu')
            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

            self.layer1 = self._make_layer(stage_index=1, layers=layers[0], in_channels=64, channels=256, stride=1)
            self.layer2 = self._make_layer(stage_index=2, layers=layers[1], in_channels=256, channels=512, stride=2)
            self.layer3 = self._make_layer(stage_index=3, layers=layers[2], in_channels=512, channels=1024, stride=2)
            self.layer4 = self._make_layer(stage_index=4, layers=layers[3], in_channels=1024, channels=2048, stride=2)

    def _make_layer(self, stage_index, layers, channels, stride, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(BottleneckV1(channels, stride, channels != in_channels, in_channels=in_channels, prefix=''))
            for _ in range(layers - 1):
                layer.add(BottleneckV1(channels, 1, False, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
