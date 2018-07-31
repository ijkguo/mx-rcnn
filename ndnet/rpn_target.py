import mxnet as mx
from mxnet import gluon
import numpy as np
from ndnet.coder import SigmoidClassEncoder, NormalizedBoxCenterEncoder


class RPNTargetSampler(gluon.Block):
    def __init__(self, num_sample, pos_iou_thresh, neg_iou_thresh, pos_ratio):
        super(RPNTargetSampler, self).__init__()
        self._num_sample = num_sample
        self._max_pos = int(round(num_sample * pos_ratio))
        self._pos_iou_thresh = pos_iou_thresh
        self._neg_iou_thresh = neg_iou_thresh
        self._eps = np.spacing(np.float32(1.0))

    def forward(self, ious):
        # ious (N, M) i.e. (num_anchors, num_gt)
        # return: matches (num_anchors,) value [0, M)
        # return: samples (num_anchors,) value 1: pos, -1: neg, 0: ignore
        matches = mx.nd.argmax(ious, axis=1)

        # samples init with 0 (ignore)
        ious_max_per_anchor = mx.nd.max(ious, axis=1)
        samples = mx.nd.zeros_like(ious_max_per_anchor)

        # set argmax (1, num_gt)
        ious_max_per_gt = mx.nd.max(ious, axis=0, keepdims=True)
        # ious (num_anchor, num_gt) >= argmax (1, num_gt) -> mark row as positive
        mask = mx.nd.broadcast_greater(ious + self._eps, ious_max_per_gt)
        # reduce column (num_anchor, num_gt) -> (num_anchor)
        mask = mx.nd.sum(mask, axis=1)
        # row maybe sampled by 2 columns but still only matches to most overlapping gt
        samples = mx.nd.where(mask, mx.nd.ones_like(samples), samples)

        # set positive overlap to 1
        samples = mx.nd.where(ious_max_per_anchor >= self._pos_iou_thresh,
                              mx.nd.ones_like(samples), samples)
        # set negative overlap to -1
        tmp = (ious_max_per_anchor < self._neg_iou_thresh) * (ious_max_per_anchor >= 0)
        samples = mx.nd.where(tmp, mx.nd.ones_like(samples) * -1, samples)

        # subsample fg labels
        samples = samples.asnumpy()
        num_pos = int((samples > 0).sum())
        if num_pos > self._max_pos:
            disable_indices = np.random.choice(
                np.where(samples > 0)[0], size=(num_pos - self._max_pos), replace=False)
            samples[disable_indices] = 0  # use 0 to ignore

        # subsample bg labels
        num_neg = int((samples < 0).sum())
        # if pos_sample is less than quota, we can have negative samples filling the gap
        max_neg = self._num_sample - min(num_pos, self._max_pos)
        if num_neg > max_neg:
            disable_indices = np.random.choice(
                np.where(samples < 0)[0], size=(num_neg - max_neg), replace=False)
            samples[disable_indices] = 0

        # convert to ndarray
        samples = mx.nd.array(samples, ctx=matches.context)
        return samples, matches


class RPNTargetGenerator:
    def __init__(self, num_sample, pos_iou_thresh, neg_iou_thresh, pos_ratio, stds):
        self._num_sample = num_sample
        self._pos_iou_thresh = pos_iou_thresh
        self._neg_iou_thresh = neg_iou_thresh
        self._pos_ratio = pos_ratio
        self._stds = stds
        self._sampler = RPNTargetSampler(num_sample, pos_iou_thresh, neg_iou_thresh, pos_ratio)
        self._cls_encoder = SigmoidClassEncoder()
        self._box_encoder = NormalizedBoxCenterEncoder(stds=stds)

    def forward(self, bbox, anchor, width, height):
        # calculate ious between (N, 4) anchors and (M, 4) bbox ground-truths
        # ious is (N, M)
        ious = mx.nd.contrib.box_iou(anchor, bbox, format='corner')

        # mask out invalid anchors, (N, 4)
        a_xmin, a_ymin, a_xmax, a_ymax = mx.nd.split(anchor, num_outputs=4, axis=-1)
        invalid_mask = (a_xmin < 0) + (a_ymin < 0) + (a_xmax >= width) + (a_ymax >= height)
        invalid_mask = mx.nd.repeat(invalid_mask, repeats=bbox.shape[0], axis=-1)
        ious = mx.nd.where(invalid_mask, mx.nd.ones_like(ious) * -1, ious)

        # matches (N) values [0, M), samples (N) values +1 pos -1 neg 0 ignore
        samples, matches = self._sampler(ious)

        # training targets for RPN
        # cls_target (N,) value (1: pos, 0: neg, -1: ignore)
        cls_target, _ = self._cls_encoder(samples)
        # box encoder is expecting (B, N), (B, N), (B, N, 4), (B, M, 4)
        # box_target, box_mask (B=1, N, 4)
        box_target, box_mask = self._box_encoder(
            samples.expand_dims(axis=0), matches.expand_dims(0), anchor.expand_dims(axis=0), bbox.expand_dims(0))
        return cls_target, box_target[0], box_mask[0]
