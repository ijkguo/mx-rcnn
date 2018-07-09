import mxnet as mx
import numpy as np
from ndnet.bbox import BBoxSplit
from ndnet.coder import SigmoidClassEncoder, NormalizedBoxCenterEncoder
from gluoncv.nn.matcher import MaximumMatcher, BipartiteMatcher, CompositeMatcher
from gluoncv.nn.sampler import QuotaSampler


class RPNTargetGenerator:
    def __init__(self, num_sample, pos_iou_thresh, neg_iou_thresh, pos_ratio, stds):
        self._num_sample = num_sample
        self._pos_iou_thresh = pos_iou_thresh
        self._neg_iou_thresh = neg_iou_thresh
        self._pos_ratio = pos_ratio
        self._stds = stds
        self._bbox_split = BBoxSplit(axis=-1)
        self._matcher = CompositeMatcher([BipartiteMatcher(), MaximumMatcher(pos_iou_thresh)])
        self._sampler = QuotaSampler(num_sample, pos_iou_thresh, neg_iou_thresh, 0., pos_ratio)
        self._cls_encoder = SigmoidClassEncoder()
        self._box_encoder = NormalizedBoxCenterEncoder(stds=stds)

    def forward(self, bbox, anchor, width, height):
        # anchor with shape (N, 4)
        a_xmin, a_ymin, a_xmax, a_ymax = self._bbox_split(anchor)

        # mask out invalid anchors, (N, 4)
        invalid_mask = ((a_xmin >= 0) * (a_ymin >= 0) * (a_xmax <= width) * (a_ymax <= height)) <= 0
        invalid_mask = mx.nd.array(np.where(invalid_mask.asnumpy() > 0)[0], ctx=anchor.context)

        # calculate ious between (N, 4) anchors and (M, 4) bbox ground-truths
        # ious is (N, M)
        ious = mx.nd.contrib.box_iou(anchor, bbox, format='corner').transpose((1, 0, 2))
        ious[:, invalid_mask, :] = -1
        matches = self._matcher(ious)
        samples = self._sampler(matches, ious)

        # training targets for RPN
        cls_target, cls_mask = self._cls_encoder(samples)
        box_target, box_mask = self._box_encoder(
            samples, matches, anchor.expand_dims(axis=0), bbox)
        return cls_target, box_target, box_mask
