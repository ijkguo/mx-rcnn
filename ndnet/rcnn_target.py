import mxnet as mx
import numpy as np
from mxnet import gluon
from gluoncv.nn.bbox import BBoxCornerToCenter
from gluoncv.utils.nn.matcher import MaximumMatcher


class MultiClassEncoder(gluon.HybridBlock):
    def __init__(self, ignore_label=-1):
        super(MultiClassEncoder, self).__init__()
        self._ignore_label = ignore_label

    def hybrid_forward(self, F, samples, matches, refs, **kwargs):
        # samples (B, N, M) (+1, -1, 0: ignore), matches (B, N) [0, M), refs (B, M)
        # reshape refs (B, M) -> (B, 1, M) -> (B, N, M)
        refs = mx.nd.repeat(refs.reshape((0, 1, -1)), axis=1, repeats=matches.shape[1])
        # ids (B, N, M) -> (B, M), note no + 1 here (processed in data pipeline)
        target_ids = mx.nd.pick(refs, matches, axis=2)
        # mark (-1: neg, 0: ignore) to -1
        targets = mx.nd.where(samples > 0.5, target_ids, mx.nd.ones_like(target_ids) * self._ignore_label)
        # mark -1 to 0: bg class
        targets = mx.nd.where(samples < -0.5, mx.nd.zeros_like(targets), targets)
        return targets


class NormalizedPerClassBoxCenterEncoder(gluon.HybridBlock):
    def __init__(self, num_class, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.)):
        super(NormalizedPerClassBoxCenterEncoder, self).__init__()
        self._num_class = num_class
        self._stds = stds
        self._means = means
        with self.name_scope():
            self._corner_to_center = BBoxCornerToCenter(split=True)

    def hybrid_forward(self, F, samples, matches, anchors, labels, refs, **kwargs):
        # refs [B, M, 4], anchors [B, N, 4], samples [B, N], matches [B, N]
        # refs [B, M, 4] -> [B, N, M, 4]
        ref_boxes = F.repeat(refs.reshape((0, 1, -1, 4)), axis=1, repeats=matches.shape[1])
        # refs [B, N, M, 4] -> 4 * [B, N, M]
        ref_boxes = F.split(ref_boxes, axis=-1, num_outputs=4, squeeze_axis=True)
        # refs 4 * [B, N, M] -> pick from matches [B, N, 1] -> concat to [B, N, 4]
        ref_boxes = F.concat(*[mx.nd.pick(ref_boxes[i], matches, axis=2).reshape((0, -1, 1)) \
                             for i in range(4)], dim=2)
        # labels [B, M] -> [B, N, M]
        ref_labels = F.repeat(labels.reshape(0, 1, -1), axis=1, repeats=matches.shape[1])
        # labels [B, N, M] -> pick from matches [B, N] -> [B, N, 1]
        ref_labels = F.pick(ref_labels, matches, axis=2).reshape((0, -1, 1))
        # transform based on x, y, w, h
        # [B, N, 4] -> 4 * [B, N] -> transform -> codecs [B, N, 4]
        g = self._corner_to_center(ref_boxes)
        a = self._corner_to_center(anchors)
        t0 = ((g[0] - a[0]) / a[2] - self._means[0]) / self._stds[0]
        t1 = ((g[1] - a[1]) / a[3] - self._means[1]) / self._stds[1]
        t2 = (F.log(g[2] / a[2]) - self._means[2]) / self._stds[2]
        t3 = (F.log(g[3] / a[3]) - self._means[3]) / self._stds[3]
        codecs = F.concat(t0, t1, t2, t3, dim=2)
        # samples [B, N] -> [B, N, 1] -> [B, N, 4] -> get +1 class
        # only the positive samples have targets
        # note that iou with padded 0 box is always 0, thus no targets
        temp = F.tile(samples.reshape((0, -1, 1)), reps=(1, 1, 4)) > 0.5
        targets = F.where(temp, codecs, mx.nd.zeros_like(codecs))
        masks = F.where(temp, mx.nd.ones_like(temp), mx.nd.zeros_like(temp))
        # expand class agnostic targets to per class targets
        out_targets = []
        out_masks = []
        for cid in range(self._num_class):
            # boolean array [B, N, 1]
            same_cid = (ref_labels == cid)
            # keep orig targets
            out_targets.append(targets)
            # but mask out the one not belong to this class [B, N, 1] -> [B, N, 4]
            out_masks.append(masks * same_cid.repeat(axis=-1, repeats=4))
        # targets, masks C * [B, N, 4] -> [C, B, N, 4] -> [B, N, C, 4]
        all_targets = F.stack(*out_targets, axis=0).transpose((1, 2, 0, 3))
        all_masks = F.stack(*out_masks, axis=0).transpose((1, 2, 0, 3))
        return all_targets, all_masks


class RCNNTargetGenerator(gluon.HybridBlock):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction, fg_overlap, box_stds, **kwargs):
        super(RCNNTargetGenerator, self).__init__(**kwargs)
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._fg_fraction = fg_fraction
        self._fg_overlap = fg_overlap
        self._box_stds = box_stds
        with self.name_scope():
            self._cls_encoder = MultiClassEncoder()
            self._box_encoder = NormalizedPerClassBoxCenterEncoder(
                num_class=num_classes, stds=box_stds, means=(0., 0., 0., 0.))
            self._maximum_matcher = MaximumMatcher(threshold=fg_overlap)

    @staticmethod
    def _sampler(matches, ious, num_sample, pos_thresh, neg_thresh_high, neg_thresh_low=0.,
                 pos_ratio=0.5, neg_ratio=0.5, fill_negative=True):
        max_pos = int(round(pos_ratio * num_sample))
        max_neg = int(neg_ratio * num_sample)

        # init with 0s, which are ignored
        result = mx.nd.zeros_like(matches)
        # negative samples with label -1
        ious_max = ious.max(axis=-1)
        neg_mask = ious_max < neg_thresh_high
        neg_mask = neg_mask * (ious_max > neg_thresh_low)
        result = mx.nd.where(neg_mask, mx.nd.ones_like(result) * -1, result)
        # positive samples
        result = mx.nd.where(matches >= 0, mx.nd.ones_like(result), result)
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
        return mx.nd.array(result, ctx=matches.context)

    def hybrid_forward(self, F, rois, gt_boxes, **kwargs):
        # slice into labels and box coordinates
        gt_labels = gt_boxes[:, :, 4:5]
        gt_boxes = gt_boxes[:, :, :4]

        new_rois = []
        new_samples = []
        new_matches = []
        for i in range(self._batch_images):
            gt_box = gt_boxes[i]
            # concat rpn roi with ground truth
            all_roi = F.concat(rois[i], gt_boxes[i], dim=0)
            # calculate ious between (N, 4) anchors and (M, 4) bbox ground-truths
            # ious is (N, M), note cannot do batch op, will get (B, N, B, M) ious
            ious = F.contrib.box_iou(all_roi, gt_box, format='corner')
            # matches (N,) coded to [0, num_classes), padded gt_boxes code to 0
            matches = self._maximum_matcher(ious)
            samples = self._sampler(matches, ious, num_sample=self._batch_rois, pos_thresh=self._fg_overlap,
                                    neg_thresh_high=self._fg_overlap, pos_ratio=self._fg_fraction)
            # slice valid samples
            sf_samples = F.where(samples == 0, F.ones_like(samples) * -999, samples)
            indices = F.argsort(sf_samples, is_ascend=False).slice_axis(axis=0, begin=0, end=self._batch_rois)
            # stack all samples together
            new_rois.append(all_roi.take(indices))
            new_samples.append(samples.take(indices))
            new_matches.append(matches.take(indices))
        new_rois = F.stack(*new_rois, axis=0)
        new_samples = F.stack(*new_samples, axis=0)
        new_matches = F.stack(*new_matches, axis=0)

        cls_target = self._cls_encoder(new_samples, new_matches, gt_labels)
        box_target, box_mask = self._box_encoder(new_samples, new_matches, new_rois, gt_labels, gt_boxes)
        return new_rois, cls_target, box_target, box_mask
