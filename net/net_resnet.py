from mxnet.gluon import nn, HybridBlock


def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)


class BottleneckV2(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm(epsilon=2e-5, use_global_stats=True)
        self.conv1 = nn.Conv2D(channels // 4, kernel_size=1, strides=1, use_bias=False)
        self.bn2 = nn.BatchNorm(epsilon=2e-5, use_global_stats=True)
        self.conv2 = _conv3x3(channels // 4, stride, channels // 4)
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


class ResNet101V2(HybridBlock):
    def __init__(self, **kwargs):
        super(ResNet101V2, self).__init__(**kwargs)
        with self.name_scope():
            self.layer0 = nn.HybridSequential(prefix='')
            self.layer0.add(nn.BatchNorm(scale=False, epsilon=2e-5, use_global_stats=True))
            self.layer0.add(nn.Conv2D(64, 7, 2, 3, use_bias=False))
            self.layer0.add(nn.BatchNorm(epsilon=2e-5, use_global_stats=True))
            self.layer0.add(nn.Activation('relu'))
            self.layer0.add(nn.MaxPool2D(3, 2, 1))

            self.layer1 = self._make_layer(stage_index=1, layers=3, in_channels=64, channels=256, stride=1)
            self.layer2 = self._make_layer(stage_index=2, layers=4, in_channels=256, channels=512, stride=2)
            self.layer3 = self._make_layer(stage_index=3, layers=23, in_channels=512, channels=1024, stride=2)
            self.layer4 = self._make_layer(stage_index=4, layers=3, in_channels=1024, channels=2048, stride=2)

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


class RPN(HybridBlock):
    def __init__(self, in_channels, num_anchors, anchor_scales, anchor_ratios,
                 rpn_feature_stride, rpn_pre_topk, rpn_post_topk, rpn_nms_thresh, rpn_min_size,
                 **kwargs):
        super(RPN, self).__init__(**kwargs)
        self._num_anchors = num_anchors
        self._anchor_scales = anchor_scales
        self._anchor_ratios = anchor_ratios
        self._rpn_feature_stride = rpn_feature_stride
        self._rpn_pre_topk = rpn_pre_topk
        self._rpn_post_topk = rpn_post_topk
        self._rpn_nms_thresh = rpn_nms_thresh
        self._rpn_min_size = rpn_min_size

        with self.name_scope():
            self.rpn_conv = nn.Conv2D(in_channels=in_channels, channels=512, kernel_size=(3, 3), padding=(1, 1))
            self.conv_cls = nn.Conv2D(in_channels=512, channels=2 * num_anchors, kernel_size=(1, 1), padding=(0, 0))
            self.conv_reg = nn.Conv2D(in_channels=512, channels=4 * num_anchors, kernel_size=(1, 1), padding=(0, 0))

    def hybrid_forward(self, F, x, im_info):
        x = F.relu(self.rpn_conv(x))
        cls = self.conv_cls(x)
        cls = F.reshape(cls, (0, 2, -1, 0))
        cls = F.softmax(cls, axis=1)
        cls = F.reshape(cls, (0, 2 * self._num_anchors, -1, 0))
        reg = self.conv_reg(x)

        rois = F.contrib.Proposal(cls_score=cls, bbox_pred=reg, im_info=im_info,
                                  rpn_pre_nms_top_n=self._rpn_pre_topk, rpn_post_nms_top_n=self._rpn_post_topk,
                                  threshold=self._rpn_nms_thresh, rpn_min_size=self._rpn_min_size,
                                  scales=self._anchor_scales, ratios=self._anchor_ratios)
        return rois


class RCNN(HybridBlock):
    def __init__(self, in_units, num_classes, **kwargs):
        super(RCNN, self).__init__(**kwargs)
        with self.name_scope():
            self.cls = nn.Dense(in_units=in_units, units=num_classes)
            self.reg = nn.Dense(in_units=in_units, units=4 * num_classes)

    def hybrid_forward(self, F, x):
        cls = self.cls(x)
        cls = F.softmax(cls)
        reg = self.reg(x)
        return cls, reg


class FRCNNResNet(HybridBlock):
    def __init__(self, num_anchors=9, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2),
                 rpn_feature_stride=16, rpn_pre_topk=6000, rpn_post_topk=300, rpn_nms_thresh=0.7, rpn_min_size=16,
                 num_classes=21, rcnn_feature_stride=16, rcnn_pooled_size=(14, 14),
                 **kwargs):
        super(FRCNNResNet, self).__init__(**kwargs)
        self._rcnn_feature_stride = rcnn_feature_stride
        self._rcnn_pooled_size = rcnn_pooled_size

        with self.name_scope():
            self.backbone = ResNet101V2(prefix='')
            self.rcnn = RCNN(2048, num_classes)
            self.rpn = RPN(1024, num_anchors, anchor_scales, anchor_ratios,
                           rpn_feature_stride, rpn_pre_topk, rpn_post_topk, rpn_nms_thresh, rpn_min_size)

    def hybrid_forward(self, F, x, im_info):
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        feat = self.backbone.layer3(x)

        rois = self.rpn(feat, im_info)
        pooled_feat = F.ROIPooling(feat, rois, self._rcnn_pooled_size, 1.0 / self._rcnn_feature_stride)
        top_feat = self.backbone.layer4(pooled_feat)
        rcnn_cls, rcnn_reg = self.rcnn(top_feat)
        return rois, rcnn_cls, rcnn_reg
