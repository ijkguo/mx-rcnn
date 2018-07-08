import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn, HybridBlock

from .proposal import Proposal
from .rcnn_target import RCNNTargetSampler, RCNNTargetGenerator
from .rcnn_inference import RCNNDetector


def get_feat_size(im_height, im_width):
    feat_shape = []
    for sz in [im_height, im_width]:
        sz = (sz + 2 * 3 - 7) // 2 + 1  # conv0 7*7 stride 2 pad 3
        sz = (sz + 2 * 1 - 3) // 2 + 1  # pool0 3*3 stride 2 pad 1
        sz = (sz + 2 * 1 - 3) // 2 + 1  # stage2 3*3 stride 2 pad 1
        sz = (sz + 2 * 1 - 3) // 2 + 1  # stage3 3*3 stride 2 pad 1
        feat_shape.append(sz)
    return feat_shape


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


class ResNet50V2(HybridBlock):
    def __init__(self, **kwargs):
        super(ResNet50V2, self).__init__(**kwargs)
        with self.name_scope():
            self.layer0 = nn.HybridSequential(prefix='')
            self.layer0.add(nn.BatchNorm(scale=False, epsilon=2e-5, use_global_stats=True))
            self.layer0.add(nn.Conv2D(64, 7, 2, 3, use_bias=False))
            self.layer0.add(nn.BatchNorm(epsilon=2e-5, use_global_stats=True))
            self.layer0.add(nn.Activation('relu'))
            self.layer0.add(nn.MaxPool2D(3, 2, 1))

            self.layer1 = self._make_layer(stage_index=1, layers=3, in_channels=64, channels=256, stride=1)
            self.layer2 = self._make_layer(stage_index=2, layers=4, in_channels=256, channels=512, stride=2)
            self.layer3 = self._make_layer(stage_index=3, layers=6, in_channels=512, channels=1024, stride=2)
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


class FRCNNResNet(HybridBlock):
    def __init__(self, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2),
                 rpn_feature_stride=16, rpn_pre_topk=6000, rpn_post_topk=300, rpn_nms_thresh=0.7, rpn_min_size=16,
                 num_classes=21, rcnn_feature_stride=16, rcnn_pooled_size=(14, 14), rcnn_batch_size=1,
                 rcnn_batch_rois=128, rcnn_fg_fraction=0.25, rcnn_fg_overlap=0.5, rcnn_bbox_stds=(0.1, 0.1, 0.2, 0.2),
                 rcnn_nms_thresh=0.3, rcnn_nms_topk=-1, rcnn_roi_mode='align', **kwargs):
        super(FRCNNResNet, self).__init__(**kwargs)
        self._num_classes = num_classes
        self._rcnn_feature_stride = rcnn_feature_stride
        self._rcnn_pooled_size = rcnn_pooled_size
        self._rcnn_batch_size = rcnn_batch_size
        self._rcnn_batch_rois = rcnn_batch_rois
        self._rcnn_roi_mode = rcnn_roi_mode

        with self.name_scope():
            self.backbone = ResNet50V2(prefix='')
            self.rcnn = RCNN(2048, num_classes)
            self.rpn = RPN(1024, len(anchor_scales) * len(anchor_ratios))
            self.proposal = Proposal(rpn_pre_topk, rpn_post_topk, rpn_nms_thresh, rpn_min_size)
            self.rcnn_sampler = RCNNTargetSampler(rcnn_batch_size, rcnn_batch_rois, rpn_post_topk, rcnn_fg_fraction, rcnn_fg_overlap)
            self.rcnn_target = RCNNTargetGenerator(rcnn_batch_rois, num_classes, rcnn_bbox_stds)
            self.rcnn_detect = RCNNDetector(rcnn_bbox_stds, num_classes, rcnn_nms_thresh, rcnn_nms_topk)

    def hybrid_forward(self, F, x, anchors, im_info, gt_boxes=None):
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        feat = self.backbone.layer3(x)

        # generate proposals
        rpn_cls, rpn_reg = self.rpn(feat, im_info)
        with autograd.pause():
            rpn_cls_prob = F.sigmoid(rpn_cls)
            rois = self.proposal(rpn_cls_prob, rpn_reg, anchors, im_info)

        # generate targets
        if autograd.is_training():
            with autograd.pause():
                rois, samples, matches = self.rcnn_sampler(rois, gt_boxes)
                rcnn_label, rcnn_bbox_target, rcnn_bbox_weight = self.rcnn_target(rois, gt_boxes, samples, matches)
                rcnn_label = F.stop_gradient(rcnn_label.reshape(-3))
                rcnn_bbox_target = F.stop_gradient(rcnn_bbox_target.reshape((-3, -3)))
                rcnn_bbox_weight = F.stop_gradient(rcnn_bbox_weight.reshape((-3, -3)))

        # create batch id and reshape for roi pooling
        with autograd.pause():
            rois = rois.reshape((-3, 0))
            roi_batch_id = F.arange(0, self._rcnn_batch_size, repeat=self._rcnn_batch_rois).reshape((-1, 1))
            rois = F.concat(roi_batch_id, rois, dim=-1)
            rois = F.stop_gradient(rois)

        # pool to roi features
        if self._rcnn_roi_mode == 'pool':
            pooled_feat = F.ROIPooling(feat, rois, self._rcnn_pooled_size, 1.0 / self._rcnn_feature_stride)
        elif self._rcnn_roi_mode == 'align':
            pooled_feat = F.contrib.ROIAlign(feat, rois, self._rcnn_pooled_size, 1.0 / self._rcnn_feature_stride)
        else:
            raise ValueError("Invalid roi mode: {}".format(self._rcnn_roi_mode))

        # classify pooled features
        top_feat = self.backbone.layer4(pooled_feat)
        rcnn_cls, rcnn_reg = self.rcnn(top_feat)

        if autograd.is_training():
            return rpn_cls, rpn_reg, rcnn_cls, rcnn_reg, rcnn_label, rcnn_bbox_target, rcnn_bbox_weight

        # rois [B, N, 4]
        rois = F.slice_axis(rois, axis=-1, begin=1, end=None)
        rois = F.reshape(rois, (self._rcnn_batch_size, self._rcnn_batch_rois, 4))

        # class id [C, N, 1]
        ids = F.arange(1, self._num_classes, repeat=self._rcnn_batch_rois)
        ids = F.reshape(ids, (self._num_classes - 1, self._rcnn_batch_rois, 1))

        # window [height, width] for clipping
        im_info = F.slice_axis(im_info, axis=-1, begin=0, end=2)

        # cls [B, C, N, 1]
        rcnn_cls = F.softmax(rcnn_cls, axis=-1)
        rcnn_cls = F.slice_axis(rcnn_cls, axis=-1, begin=1, end=None)
        rcnn_cls = F.reshape(rcnn_cls, (self._rcnn_batch_size, self._rcnn_batch_rois, self._num_classes - 1, 1))
        rcnn_cls = F.transpose(rcnn_cls, (0, 2, 1, 3))

        # reg [B, C, N, 4]
        rcnn_reg = F.slice_axis(rcnn_reg, axis=-1, begin=4, end=None)
        rcnn_reg = F.reshape(rcnn_reg, (self._rcnn_batch_size, self._rcnn_batch_rois, self._num_classes - 1, 4))
        rcnn_reg = F.transpose(rcnn_reg, (0, 2, 1, 3))

        ret_ids = []
        ret_scores = []
        ret_bboxes = []
        for i in range(self._rcnn_batch_size):
            b_rois = F.squeeze(F.slice_axis(rois, axis=0, begin=i, end=i+1), axis=0)
            b_cls = F.squeeze(F.slice_axis(rcnn_cls, axis=0, begin=i, end=i+1), axis=0)
            b_reg = F.squeeze(F.slice_axis(rcnn_reg, axis=0, begin=i, end=i+1), axis=0)
            b_im_info = F.squeeze(F.slice_axis(im_info, axis=0, begin=i, end=i+1), axis=0)
            scores, bboxes = self.rcnn_detect(b_rois, b_cls, b_reg, b_im_info)
            b_ids = F.where(scores < 0, F.ones_like(ids) * -1, ids)
            ret_ids.append(b_ids.reshape((-1, 1)))
            ret_scores.append(scores.reshape((-1, 1)))
            ret_bboxes.append(bboxes.reshape((-1, 4)))
        ret_ids = F.stack(*ret_ids, axis=0)
        ret_scores = F.stack(*ret_scores, axis=0)
        ret_bboxes = F.stack(*ret_bboxes, axis=0)
        return ret_ids, ret_scores, ret_bboxes
