from mxnet import gluon
from ndnet.coder import NormalizedPerClassBoxCenterEncoder, MultiClassEncoder


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
