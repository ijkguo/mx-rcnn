import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn, HybridBlock

from symdata.anchor import AnchorGenerator
from nddata.transform import batchify_append, batchify_pad, split_append, split_pad
from .rpn_target import RPNTargetGenerator
from .rpn_inference import Proposal
from .rcnn_target import RCNNTargetSampler, RCNNTargetGenerator
from .rcnn_inference import RCNNDetector


class RPN(HybridBlock):
    def __init__(self, rpn_channels, num_anchors, **kwargs):
        super(RPN, self).__init__(**kwargs)
        self._num_anchors = num_anchors

        weight_initializer = mx.initializer.Normal(0.01)
        with self.name_scope():
            self.rpn_conv = nn.Conv2D(channels=rpn_channels, kernel_size=(3, 3), padding=(1, 1), weight_initializer=weight_initializer)
            self.conv_cls = nn.Conv2D(channels=num_anchors, kernel_size=(1, 1), padding=(0, 0), weight_initializer=weight_initializer)
            self.conv_reg = nn.Conv2D(channels=4 * num_anchors, kernel_size=(1, 1), padding=(0, 0), weight_initializer=weight_initializer)

    def hybrid_forward(self, F, x, im_info):
        x = F.relu(self.rpn_conv(x))
        # (B, N, H, W) -> (B, H * W * N, 1)
        cls = self.conv_cls(x).transpose((0, 2, 3, 1)).reshape((0, -1, 1))
        # (B, N * 4, H, W) -> (B, H * W * N, 4)
        reg = self.conv_reg(x).transpose((0, 2, 3, 1)).reshape((0, -1, 4))
        return cls, reg


class RCNN(HybridBlock):
    def __init__(self, batch_images, num_classes, **kwargs):
        super(RCNN, self).__init__(**kwargs)
        self._batch_images = batch_images
        self._num_classes = num_classes
        with self.name_scope():
            self.global_avg_pool = nn.GlobalAvgPool2D()
            self.cls = nn.Dense(units=num_classes, weight_initializer=mx.initializer.Normal(0.01))
            self.reg = nn.Dense(units=4 * num_classes, weight_initializer=mx.initializer.Normal(0.001))

    def hybrid_forward(self, F, x):
        # (B * N, channels, 7, 7) -> (B * N, channels)
        x = self.global_avg_pool(x)
        # (B * N, C) -> (B, N, C)
        cls = self.cls(x).reshape((self._batch_images, -1, self._num_classes))
        # (B * N, C * 4) -> (B, N, C, 4)
        reg = self.reg(x).reshape((self._batch_images, -1, self._num_classes, 4))
        return cls, reg


class FRCNN(HybridBlock):
    def __init__(self, features, top_features, batch_images=1, train_patterns=None,
                 img_short=600, img_max_size=1000, img_means=(0., 0., 0.), img_stds=(0., 0., 0.), clip=4.42,
                 rpn_feature_stride=16, rpn_anchor_scales=(8, 16, 32), rpn_anchor_ratios=(0.5, 1, 2),
                 rpn_channels=1024, rpn_nms_thresh=0.7, rpn_min_size=16,
                 rpn_train_pre_topk=12000, rpn_train_post_topk=2000, rpn_test_pre_topk=6000, rpn_test_post_topk=300,
                 rcnn_feature_stride=16, rcnn_pooled_size=(14, 14), rcnn_roi_mode='align',
                 rcnn_num_classes=21, rcnn_batch_rois=128, rcnn_bbox_stds=(0.1, 0.1, 0.2, 0.2),
                 rpn_batch_rois=256, rpn_fg_overlap=0.7, rpn_bg_overlap=0.3, rpn_fg_fraction=0.5,  # only used for RPNTarget
                 rcnn_fg_fraction=0.25, rcnn_fg_overlap=0.5,  # only used for RCNNTarget
                 rcnn_nms_thresh=0.3, rcnn_nms_topk=-1,  # only used for RCNN inference
                 **kwargs):
        super(FRCNN, self).__init__(**kwargs)
        self._rcnn_feature_stride = rcnn_feature_stride
        self._rcnn_roi_mode = rcnn_roi_mode
        self._rcnn_pooled_size = rcnn_pooled_size
        self._rcnn_num_classes = rcnn_num_classes
        self._batch_images = batch_images
        self._rpn_test_post_topk = rpn_test_post_topk
        self._rcnn_batch_rois = rcnn_batch_rois

        self.img_short = img_short
        self.img_max_size = img_max_size
        self.img_means = img_means
        self.img_stds = img_stds
        self.train_patterns = train_patterns
        ag = AnchorGenerator(
            feat_stride=rpn_feature_stride, anchor_scales=rpn_anchor_scales, anchor_ratios=rpn_anchor_ratios)
        alloc_size = int(img_max_size * 1.5 / rpn_feature_stride)
        self.anchors = mx.nd.array(ag.generate(alloc_size, alloc_size)).reshape((alloc_size, alloc_size, -1))
        self.anchor_target = RPNTargetGenerator(
            num_sample=rpn_batch_rois, pos_iou_thresh=rpn_fg_overlap, neg_iou_thresh=rpn_bg_overlap,
            pos_ratio=rpn_fg_fraction, stds=(1.0, 1.0, 1.0, 1.0))
        self.batchify_fn = batchify_append if batch_images == 1 else batchify_pad
        self.split_fn = split_append if batch_images == 1 else split_pad

        self.features = features
        self.top_features = top_features
        self.rpn = RPN(rpn_channels, len(rpn_anchor_scales) * len(rpn_anchor_ratios))
        self.rcnn = RCNN(batch_images, rcnn_num_classes)
        self.proposal = Proposal(clip, rpn_nms_thresh, rpn_min_size,
            rpn_train_pre_topk, rpn_train_post_topk, rpn_test_pre_topk, rpn_test_post_topk)
        self.rcnn_sampler = RCNNTargetSampler(batch_images, rcnn_batch_rois, rpn_train_post_topk, rcnn_fg_fraction, rcnn_fg_overlap)
        self.rcnn_target = RCNNTargetGenerator(rcnn_batch_rois, rcnn_num_classes, rcnn_bbox_stds)
        self.rcnn_detect = RCNNDetector(clip, rcnn_bbox_stds, rcnn_num_classes, rcnn_nms_thresh, rcnn_nms_topk)

    def anchor_shape_fn(self, im_height, im_width):
        feat_sym = self.features(mx.sym.var(name='data'))
        _, oshape, _ = feat_sym.infer_shape(data=(1, 3, im_height, im_width))
        return oshape[0][-2:]

    def fastrcnn_inference(self, F, rois, rcnn_cls, rcnn_reg, im_info, num_rois):
        # rois [B, N, 4] rcnn_cls [B, N, C], rcnn_reg [B, N, C, 4] im_info [B, 3]
        def _split(x, axis, num_outputs, squeeze_axis):
            x = F.split(x, axis=axis, num_outputs=num_outputs, squeeze_axis=squeeze_axis)
            if isinstance(x, list):
                return x
            else:
                return [x]

        # class id [C, N, 1]
        ids = F.arange(1, self._rcnn_num_classes, repeat=num_rois)
        ids = F.reshape(ids, (self._rcnn_num_classes - 1, num_rois, 1))

        # window [height, width] for clipping
        im_info = F.slice_axis(im_info, axis=-1, begin=0, end=2)

        # cls [B, N, C] -> [B, N, C - 1] -> [B, C - 1, N, 1]
        rcnn_cls = F.softmax(rcnn_cls, axis=-1)
        rcnn_cls = F.slice_axis(rcnn_cls, axis=-1, begin=1, end=None)
        rcnn_cls = F.reshape(rcnn_cls, (0, 0, 0, 1))
        rcnn_cls = F.transpose(rcnn_cls, (0, 2, 1, 3))

        # reg [B, N, C, 4] -> [B, N, C - 1, 4] -> [B, C - 1, N, 4]
        rcnn_reg = F.slice_axis(rcnn_reg, axis=-2, begin=1, end=None)
        rcnn_reg = F.transpose(rcnn_reg, (0, 2, 1, 3))

        # rois (B, N, 4) -> B * (1, N, 4)
        rcnn_rois = _split(rois, axis=0, num_outputs=self._batch_images, squeeze_axis=False)
        # cls (B, C - 1, N, 1) -> B * (C - 1, N, 1)
        rcnn_cls = _split(rcnn_cls, axis=0, num_outputs=self._batch_images, squeeze_axis=True)
        # reg (B, C - 1, N, 4) -> B * (C - 1, N, 4)
        rcnn_reg = _split(rcnn_reg, axis=0, num_outputs=self._batch_images, squeeze_axis=True)
        # im_info (B, 2) -> B * (2,)
        im_info = _split(im_info, axis=0, num_outputs=self._batch_images, squeeze_axis=True)

        # per batch predict, nms, each class has topk outputs
        ret_ids, ret_scores, ret_bboxes = [], [], []
        for rois, cls, reg, window in zip(rcnn_rois, rcnn_cls, rcnn_reg, im_info):
            scores, bboxes = self.rcnn_detect(rois, cls, reg, window)
            ids = F.where(scores < 0, F.ones_like(ids) * -1, ids)
            ret_ids.append(ids)
            ret_scores.append(scores)
            ret_bboxes.append(bboxes)

        # ret_ids [B, C, topk, 1], ret_scores [B, C, topk, 1], ret_bboxes [B, C, topk, 4]
        ret_ids = F.stack(*ret_ids, axis=0)
        ret_scores = F.stack(*ret_scores, axis=0)
        ret_bboxes = F.stack(*ret_bboxes, axis=0)
        return ret_ids, ret_scores, ret_bboxes

    def hybrid_forward(self, F, x, anchors, im_info, gt_boxes=None):
        feat = self.features(x)

        # generate proposals
        rpn_cls, rpn_reg = self.rpn(feat, im_info)
        rpn_cls_prob, rpn_reg_out = F.sigmoid(F.stop_gradient(rpn_cls)), F.stop_gradient(rpn_reg)
        rois, scores = self.proposal(rpn_cls_prob, rpn_reg_out, anchors, im_info)
        rois, scores = F.stop_gradient(rois), F.stop_gradient(scores)

        # generate targets
        if autograd.is_training():
            rois, samples, matches = self.rcnn_sampler(rois, scores, gt_boxes)
            rcnn_label, rcnn_bbox_target, rcnn_bbox_weight = self.rcnn_target(rois, gt_boxes, samples, matches)

        # create batch id and reshape for roi pooling
        num_rois = self._rcnn_batch_rois if autograd.is_training() else self._rpn_test_post_topk
        roi_batch_id = F.arange(0, self._batch_images, repeat=num_rois)
        padded_rois = F.concat(rois.reshape((-3, 0)), roi_batch_id.reshape((-1, 1)), dim=-1)
        padded_rois = F.stop_gradient(padded_rois)

        # pool to roi features
        if self._rcnn_roi_mode == 'pool':
            pooled_feat = F.ROIPooling(feat, padded_rois, self._rcnn_pooled_size, 1.0 / self._rcnn_feature_stride)
        elif self._rcnn_roi_mode == 'align':
            pooled_feat = F.contrib.ROIAlign(feat, padded_rois, self._rcnn_pooled_size, 1.0 / self._rcnn_feature_stride, sample_ratio=2)
        else:
            raise ValueError("Invalid roi mode: {}".format(self._rcnn_roi_mode))

        # classify pooled features
        top_feat = self.top_features(pooled_feat)
        rcnn_cls, rcnn_reg = self.rcnn(top_feat)

        if autograd.is_training():
            return rpn_cls, rpn_reg, rcnn_cls, rcnn_reg, rcnn_label, rcnn_bbox_target, rcnn_bbox_weight
        return self.fastrcnn_inference(F, rois, rcnn_cls, rcnn_reg, im_info, num_rois)
