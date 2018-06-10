import mxnet as mx
import numpy as np

from .np_anchor import AnchorGenerator


# generate and clip anchor
class RPNAnchorGenerator:
    def __init__(self, ag: AnchorGenerator, alloc_size=(128, 128)):
        self._anchors = mx.nd.array(ag.generate(*alloc_size).reshape((1, 1, alloc_size[0], alloc_size[1], -1)))

    def forward(self, feat_height, feat_width):
        return self._anchors[:, :, :feat_height, :feat_width, :].reshape(-1, 4)


class RPNTargetGenerator:
    def __init__(self, num_sample, pos_iou_thresh, neg_iou_thresh, pos_ratio, stds):
        self._num_sample = num_sample
        self._pos_iou_thresh = pos_iou_thresh
        self._neg_iou_thresh = neg_iou_thresh
        self._pos_ratio = pos_ratio
        self._stds = stds

    @staticmethod
    def _bbox_split(x):
        return mx.nd.split(x, axis=-1, num_outputs=4)

    @staticmethod
    def _maximum_matcher(x, threshold):
        # maximum matching based on threshold
        argmax = mx.nd.argmax(x, axis=-1)
        match = mx.nd.where(mx.nd.pick(x, argmax, axis=-1) > threshold, argmax,
                            mx.nd.ones_like(argmax) * -1)
        return match

    @staticmethod
    def _bipartite_matcher_v1(x, threshold=1e-12, is_ascend=False, eps=1e-12):
        # bipartite matching strategy
        match = mx.nd.contrib.bipartite_matching(x, threshold=threshold, is_ascend=is_ascend)
        # make sure if iou(a, y) == iou(b, y), then b should also be a good match
        # otherwise positive/negative samples are confusing
        # potential argmax and max
        pargmax = x.argmax(axis=-1, keepdims=True)  # (B, num_anchor, 1)
        maxs = x.max(axis=-2, keepdims=True)  # (B, 1, num_gt)
        pmax = mx.nd.pick(x, pargmax, axis=-1, keepdims=True)  # (B, num_anchor, 1)
        mask = mx.nd.broadcast_greater_equal(pmax + eps, maxs)  # (B, num_anchor, num_gt)
        mask = mx.nd.pick(mask, pargmax, axis=-1, keepdims=True)  # (B, num_anchor, 1)
        new_match = mx.nd.where(mask > 0, pargmax, mx.nd.ones_like(pargmax) * -1)
        result = mx.nd.where(match[0] < 0, new_match.squeeze(axis=-1), match[0])
        return result

    @staticmethod
    def _matcher(x, pos_iou_thresh):
        matches = [RPNTargetGenerator._maximum_matcher(x, pos_iou_thresh),
                   RPNTargetGenerator._bipartite_matcher_v1(x)]
        result = matches[0]
        for match in matches[1:]:
            result = mx.nd.where(result > -0.5, result, match)
        return result

    @staticmethod
    def _sampler(matches, ious, num_sample, pos_thresh, neg_thresh_high, neg_thresh_low=0.,
                 pos_ratio=0.5, neg_ratio=0.5, fill_negative=True):
        max_pos = int(round(pos_ratio * num_sample))
        max_neg = int(neg_ratio * num_sample)
        results = []
        for i in range(matches.shape[0]):
            # init with 0s, which are ignored
            result = mx.nd.zeros_like(matches[0])
            # negative samples with label -1
            ious_max = ious.max(axis=-1)[i]
            neg_mask = ious_max < neg_thresh_high
            neg_mask = neg_mask * (ious_max > neg_thresh_low)
            result = mx.nd.where(neg_mask, mx.nd.ones_like(result) * -1, result)
            # positive samples
            result = mx.nd.where(matches[i] >= 0, mx.nd.ones_like(result), result)
            result = mx.nd.where(ious_max >= pos_thresh, mx.nd.ones_like(result), result)

            # re-balance if number of postive or negative exceed limits
            result = result.asnumpy()
            num_pos = int((result > 0).sum())
            if num_pos > max_pos:
                disable_indices = np.random.choice(
                    np.where(result > 0)[0], size=(num_pos - max_pos), replace=False)
                result[disable_indices] = 0  # use 0 to ignore
            num_neg = int((result < 0).sum())
            if fill_negative:
                # if pos_sample is less than quota, we can have negative samples filling the gap
                max_neg = max(num_sample - min(num_pos, max_pos), max_neg)
            if num_neg > max_neg:
                disable_indices = np.random.choice(
                    np.where(result < 0)[0], size=(num_neg - max_neg), replace=False)
                result[disable_indices] = 0
            results.append(mx.nd.array(result))
        return mx.nd.stack(*results, axis=0)

    @staticmethod
    def _cls_encoder(samples):
        # notation from samples, 1:pos, 0:ignore, -1:negative
        target = (samples + 1) / 2.
        target = mx.nd.where(mx.nd.abs(samples) < 1e-5, mx.nd.ones_like(target) * -1, target)
        # output: 1: pos, 0: negative, -1: ignore
        mask = mx.nd.where(mx.nd.abs(samples) > 1e-5, mx.nd.ones_like(samples), mx.nd.zeros_like(samples))
        return target, mask

    @staticmethod
    def _corner_to_center(x, axis=-1, split=True):
        xmin, ymin, xmax, ymax = mx.nd.split(x, axis=axis, num_outputs=4)
        width = xmax - xmin
        height = ymax - ymin
        x = xmin + width / 2
        y = ymin + height / 2
        if not split:
            return mx.nd.concat(x, y, width, height, dim=axis)
        else:
            return x, y, width, height

    @staticmethod
    def _box_encoder(samples, matches, anchors, refs,
                     stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.)):
        ref_boxes = mx.nd.repeat(refs.reshape((0, 1, -1, 4)), axis=1, repeats=matches.shape[1])
        ref_boxes = mx.nd.split(ref_boxes, axis=-1, num_outputs=4, squeeze_axis=True)
        ref_boxes = mx.nd.concat(*[mx.nd.pick(ref_boxes[i], matches, axis=2).reshape((0, -1, 1)) \
                                   for i in range(4)], dim=2)
        g = RPNTargetGenerator._corner_to_center(ref_boxes)
        a = RPNTargetGenerator._corner_to_center(anchors)
        t0 = ((g[0] - a[0]) / a[2] - means[0]) / stds[0]
        t1 = ((g[1] - a[1]) / a[3] - means[1]) / stds[1]
        t2 = (mx.nd.log(g[2] / a[2]) - means[2]) / stds[2]
        t3 = (mx.nd.log(g[3] / a[3]) - means[3]) / stds[3]
        codecs = mx.nd.concat(t0, t1, t2, t3, dim=2)
        temp = mx.nd.tile(samples.reshape((0, -1, 1)), reps=(1, 1, 4)) > 0.5
        targets = mx.nd.where(temp, codecs, mx.nd.zeros_like(codecs))
        masks = mx.nd.where(temp, mx.nd.ones_like(temp), mx.nd.zeros_like(temp))
        return targets, masks

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
        matches = self._matcher(ious, self._pos_iou_thresh)
        samples = self._sampler(matches, ious, self._num_sample, self._pos_iou_thresh,
                                self._neg_iou_thresh, 0., self._pos_ratio)

        # training targets for RPN
        cls_target, cls_mask = self._cls_encoder(samples)
        box_target, box_mask = self._box_encoder(
            samples, matches, anchor.expand_dims(axis=0), bbox, stds=self._stds)
        return cls_target, box_target, box_mask
