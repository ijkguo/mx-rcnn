from mxnet import gluon
from gluoncv.nn.bbox import BBoxCornerToCenter


class QuotaSampler(gluon.HybridBlock):
    def __init__(self, num_sample, num_input, pos_ratio, pos_thresh):
        super(QuotaSampler, self).__init__()
        self._num_sample = num_sample
        self._num_input = num_input
        self._max_pos = int(round(num_sample * pos_ratio))
        self._pos_thresh = pos_thresh

    def hybrid_forward(self, F, ious, **kwargs):
        # ious (N, M)
        # return: indices (num_samples,) row index to ious
        # return: samples (num_samples,) value 1: pos, 0: neg
        # return: matches (num_samples,) value [0, M)
        ious_max = ious.max(axis=-1)
        ious_argmax = ious.argmax(axis=-1)
        # init with 0, which are neg samples
        mask = F.zeros_like(ious_max)
        # positive samples
        pos_mask = ious_max >= self._pos_thresh
        mask = F.where(pos_mask, F.ones_like(mask), mask)

        # shuffle mask
        rand = F.random.uniform(0, 1, shape=(self._num_input,))
        index = F.argsort(rand)
        mask = F.take(mask, index)
        ious_argmax = F.take(ious_argmax, index)

        # sample pos and neg samples
        order = F.argsort(mask, is_ascend=False)
        topk = F.slice_axis(order, axis=0, begin=0, end=self._max_pos)
        bottomk = F.slice_axis(order, axis=0, begin=-(self._num_sample - self._max_pos), end=None)
        selected = F.concat(topk, bottomk, dim=0)

        # output
        indices = F.take(index, selected)
        samples = F.take(mask, selected)
        matches = F.take(ious_argmax, selected)
        return indices, samples, matches


class MultiClassEncoder(gluon.HybridBlock):
    def __init__(self, num_sample, ignore_label=-1):
        super(MultiClassEncoder, self).__init__()
        self._num_sample = num_sample
        self._ignore_label = ignore_label

    def hybrid_forward(self, F, samples, matches, refs, **kwargs):
        # samples (B, N, M) (+1, -1, 0: ignore), matches (B, N) [0, M), refs (B, M)
        # reshape refs (B, M) -> (B, 1, M) -> (B, N, M)
        refs = F.repeat(refs.reshape((0, 1, -1)), axis=1, repeats=self._num_sample)
        # ids (B, N, M) -> (B, M), note no + 1 here (processed in data pipeline)
        target_ids = F.pick(refs, matches, axis=2)
        # samples 1/0, mask out neg samples to 0
        targets = F.where(samples > 0.5, target_ids, F.zeros_like(target_ids))
        return targets


class NormalizedPerClassBoxCenterEncoder(gluon.HybridBlock):
    def __init__(self, num_class, num_sample, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.)):
        super(NormalizedPerClassBoxCenterEncoder, self).__init__()
        self._num_class = num_class
        self._num_sample = num_sample
        self._stds = stds
        self._means = means
        with self.name_scope():
            self._corner_to_center = BBoxCornerToCenter(split=True)

    def hybrid_forward(self, F, samples, matches, anchors, labels, refs, **kwargs):
        # refs [B, M, 4], anchors [B, N, 4], samples [B, N], matches [B, N]
        # refs [B, M, 4] -> reshape [B, 1, M, 4] -> repeat [B, N, M, 4]
        ref_boxes = F.repeat(refs.reshape((0, 1, -1, 4)), axis=1, repeats=self._num_sample)
        # refs [B, N, M, 4] -> 4 * [B, N, M]
        ref_boxes = F.split(ref_boxes, axis=-1, num_outputs=4, squeeze_axis=True)
        # refs 4 * [B, N, M] -> pick from matches [B, N, 1] -> concat to [B, N, 4]
        ref_boxes = F.concat(*[F.pick(ref_boxes[i], matches, axis=2).reshape((0, -1, 1)) \
                             for i in range(4)], dim=2)
        # labels [B, M] -> [B, N, M]
        ref_labels = F.repeat(labels.reshape((0, 1, -1)), axis=1, repeats=self._num_sample)
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
        targets = F.where(temp, codecs, F.zeros_like(codecs))
        masks = F.where(temp, F.ones_like(temp), F.zeros_like(temp))
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


class RCNNTargetSampler(gluon.HybridBlock):
    def __init__(self, batch_images, batch_rois, batch_proposals, fg_fraction, fg_overlap, **kwargs):
        super(RCNNTargetSampler, self).__init__(**kwargs)
        self._batch_images = batch_images
        with self.name_scope():
            self._sampler = QuotaSampler(
                num_sample=batch_rois, num_input=batch_proposals, pos_ratio=fg_fraction, pos_thresh=fg_overlap)

    def hybrid_forward(self, F, rois, gt_boxes, **kwargs):
        # slice into box coordinates
        gt_boxes = F.slice_axis(gt_boxes, axis=-1, begin=0, end=4)

        # collect results into list
        new_rois = []
        new_samples = []
        new_matches = []
        for i in range(self._batch_images):
            roi = F.squeeze(F.slice_axis(rois, axis=0, begin=i, end=i+1), axis=0)
            gt_box = F.squeeze(F.slice_axis(gt_boxes, axis=0, begin=i, end=i+1), axis=0)
            # concat rpn roi with ground truth
            all_roi = F.concat(roi, gt_box, dim=0)
            # calculate (N, M) ious between (N, 4) anchors and (M, 4) bbox ground-truths
            # NOTE cannot do batch op, will get (B, N, B, M) ious
            ious = F.contrib.box_iou(all_roi, gt_box, format='corner')
            # matches (N,) coded to [0, num_classes), padded gt_boxes code to 0
            indices, samples, matches = self._sampler(ious)
            # stack all samples together
            new_rois.append(all_roi.take(indices))
            new_samples.append(samples)
            new_matches.append(matches)
        new_rois = F.stack(*new_rois, axis=0)
        new_samples = F.stack(*new_samples, axis=0)
        new_matches = F.stack(*new_matches, axis=0)
        return new_rois, new_samples, new_matches


class RCNNTargetGenerator(gluon.HybridBlock):
    def __init__(self, batch_rois, num_classes, box_stds, **kwargs):
        super(RCNNTargetGenerator, self).__init__(**kwargs)
        with self.name_scope():
            self._cls_encoder = MultiClassEncoder(num_sample=batch_rois)
            self._box_encoder = NormalizedPerClassBoxCenterEncoder(
                num_class=num_classes, num_sample=batch_rois, stds=box_stds, means=(0., 0., 0., 0.))

    def hybrid_forward(self, F, rois, gt_boxes, samples, matches, **kwargs):
        # slice into labels and box coordinates
        gt_labels = F.slice_axis(gt_boxes, axis=-1, begin=4, end=5)
        gt_boxes = F.slice_axis(gt_boxes, axis=-1, begin=0, end=4)

        cls_target = self._cls_encoder(samples, matches, gt_labels)
        box_target, box_mask = self._box_encoder(samples, matches, rois, gt_labels, gt_boxes)
        return cls_target, box_target, box_mask
