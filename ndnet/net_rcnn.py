import mxnet as mx
from mxnet.gluon import nn, HybridBlock


class RPN(HybridBlock):
    def __init__(self, in_channels, num_anchors, **kwargs):
        super(RPN, self).__init__(**kwargs)
        self._num_anchors = num_anchors

        weight_initializer = mx.initializer.Normal(0.01)
        with self.name_scope():
            self.rpn_conv = nn.Conv2D(in_channels=in_channels, channels=1024, kernel_size=(3, 3), padding=(1, 1), weight_initializer=weight_initializer)
            self.conv_cls = nn.Conv2D(in_channels=1024, channels=num_anchors, kernel_size=(1, 1), padding=(0, 0), weight_initializer=weight_initializer)
            self.conv_reg = nn.Conv2D(in_channels=1024, channels=4 * num_anchors, kernel_size=(1, 1), padding=(0, 0), weight_initializer=weight_initializer)

    def hybrid_forward(self, F, x, im_info):
        x = F.relu(self.rpn_conv(x))
        cls = self.conv_cls(x)
        reg = self.conv_reg(x)
        return cls, reg


class RCNN(HybridBlock):
    def __init__(self, in_units, num_classes, **kwargs):
        super(RCNN, self).__init__(**kwargs)
        with self.name_scope():
            self.cls = nn.Dense(in_units=in_units, units=num_classes, weight_initializer=mx.initializer.Normal(0.01))
            self.reg = nn.Dense(in_units=in_units, units=4 * num_classes, weight_initializer=mx.initializer.Normal(0.001))

    def hybrid_forward(self, F, x):
        cls = self.cls(x)
        reg = self.reg(x)
        return cls, reg


class FRCNN(HybridBlock):
    def __init__(self, **kwargs):
        super(FRCNN, self).__init__(**kwargs)
        self.anchor_generator = None
        self.anchor_shape_fn = None
        self.anchor_target = None
        self.batchify_fn = None
        self.split_fn = None

    def hybrid_forward(self, F, x, *args, **kwargs):
        raise NotImplementedError
