from mxnet import gluon
from ndnet.coder import NormalizedPerClassBoxCenterEncoder, MultiClassEncoder


class RCNNTargetSampler(gluon.HybridBlock):
    def __init__(self, num_image, num_sample, num_proposal, pos_ratio, pos_iou_thresh, **kwargs):
        super(RCNNTargetSampler, self).__init__(**kwargs)
        self._num_image = num_image
        self._num_sample = num_sample
        self._num_proposal = num_proposal
        self._max_pos = int(round(num_sample * pos_ratio))
        self._pos_iou_thresh = pos_iou_thresh

    def hybrid_forward(self, F, rois, scores, gt_boxes, **kwargs):
        # slice into box coordinates
        gt_boxes = F.slice_axis(gt_boxes, axis=-1, begin=0, end=4)

        # collect results into list
        new_rois = []
        new_samples = []
        new_matches = []
        for i in range(self._num_image):
            roi = F.squeeze(F.slice_axis(rois, axis=0, begin=i, end=i+1), axis=0)
            score = F.squeeze(F.slice_axis(scores, axis=0, begin=i, end=i+1), axis=0)
            gt_box = F.squeeze(F.slice_axis(gt_boxes, axis=0, begin=i, end=i+1), axis=0)
            gt_score = F.ones_like(F.sum(gt_box, axis=-1, keepdims=True))

            # concat rpn roi with ground truth
            all_roi = F.concat(roi, gt_box, dim=0)
            all_score = F.concat(score, gt_score, dim=0).squeeze(axis=-1)
            # calculate (N, M) ious between (N, 4) anchors and (M, 4) bbox ground-truths
            # NOTE cannot do batch op, will get (B, N, B, M) ious
            ious = F.contrib.box_iou(all_roi, gt_box, format='corner')

            # match to argmax iou
            ious_max = ious.max(axis=-1)
            ious_argmax = ious.argmax(axis=-1)
            # init with 2, which are neg samples
            mask = F.ones_like(ious_max) * 2
            # mark all ignore to 0
            mask = F.where(all_score < 0, F.zeros_like(mask), mask)
            # mark positive samples with 3
            pos_mask = ious_max >= self._pos_iou_thresh
            mask = F.where(pos_mask, F.ones_like(mask) * 3, mask)

            # shuffle mask
            rand = F.random.uniform(0, 1, shape=(self._num_proposal + 100,))
            rand = F.slice_like(rand, ious_argmax)
            index = F.argsort(rand)
            mask = F.take(mask, index)
            ious_argmax = F.take(ious_argmax, index)

            # sample pos samples
            order = F.argsort(mask, is_ascend=False)
            topk = F.slice_axis(order, axis=0, begin=0, end=self._max_pos)
            topk_indices = F.take(index, topk)
            topk_samples = F.take(mask, topk)
            topk_matches = F.take(ious_argmax, topk)
            # reset output: 3 pos 2 neg 0 ignore -> 1 pos -1 neg 0 ignore
            topk_samples = F.where(topk_samples == 3, F.ones_like(topk_samples), topk_samples)
            topk_samples = F.where(topk_samples == 2, F.ones_like(topk_samples) * -1, topk_samples)

            # sample neg samples
            index = F.slice_axis(index, axis=0, begin=self._max_pos, end=None)
            mask = F.slice_axis(mask, axis=0, begin=self._max_pos, end=None)
            ious_argmax = F.slice_axis(ious_argmax, axis=0, begin=self._max_pos, end=None)
            # change mask: 4 neg 3 pos 0 ignore
            mask = F.where(mask == 2, F.ones_like(mask) * 4, mask)
            order = F.argsort(mask, is_ascend=False)
            num_neg = self._num_sample - self._max_pos
            bottomk = F.slice_axis(order, axis=0, begin=0, end=num_neg)
            bottomk_indices = F.take(index, bottomk)
            bottomk_samples = F.take(mask, bottomk)
            bottomk_matches = F.take(ious_argmax, bottomk)
            # reset output: 4 neg 3 pos 0 ignore -> 1 pos -1 neg 0 ignore
            bottomk_samples = F.where(bottomk_samples == 3, F.ones_like(bottomk_samples), bottomk_samples)
            bottomk_samples = F.where(bottomk_samples == 4, F.ones_like(bottomk_samples) * -1, bottomk_samples)

            # output
            # indices (num_samples,) row index to ious
            indices = F.concat(topk_indices, bottomk_indices, dim=0)
            # samples (num_samples,) value 1: pos, 0: ignore, -1: neg
            samples = F.concat(topk_samples, bottomk_samples, dim=0)
            # matches (num_samples,) value [0, M)
            matches = F.concat(topk_matches, bottomk_matches, dim=0)

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

        # cls_target (B, N)
        cls_target = self._cls_encoder(samples, matches, gt_labels)
        # box_target, box_weight (C, B, N, 4)
        box_target, box_mask = self._box_encoder(samples, matches, rois, gt_labels, gt_boxes)
        # modify shapes (C, B, N, 4) -> (B, N, C, 4)
        box_target = box_target.transpose((1, 2, 0, 3))
        box_mask = box_mask.transpose((1, 2, 0, 3))
        return cls_target, box_target, box_mask


class MaskTargetGenerator(gluon.HybridBlock):
    def __init__(self, num_images, num_rois, num_classes, mask_size, **kwargs):
        self._num_images = num_images
        self._num_rois = num_rois
        self._num_classes = num_classes
        self._mask_size = mask_size
        super(MaskTargetGenerator, self).__init__(**kwargs)

    def hybrid_forward(self, F, rois, gt_masks, matches, cls_targets, **kwargs):
        # cannot know M (num_gt) to have accurate batch id B * M, must split batch dim
        def _split(x, axis, num_outputs, squeeze_axis):
            x = F.split(x, axis=axis, num_outputs=num_outputs, squeeze_axis=squeeze_axis)
            if isinstance(x, list):
                return x
            else:
                return [x]

        # gt_masks (B, M, H, W) -> (B, M, 1, H, W) -> B * (M, 1, H, W)
        gt_masks = gt_masks.reshape((0, -4, -1, 1, 0, 0))
        gt_masks = _split(gt_masks, axis=0, num_outputs=self._num_images, squeeze_axis=True)
        # rois (B, N, 4) -> B * (N, 4)
        rois = _split(rois, axis=0, num_outputs=self._num_images, squeeze_axis=True)
        # remove possible -1 match
        matches = F.where(matches >= 0, matches, F.zeros_like(matches))
        # matches (B, N) -> B * (N,)
        matches = _split(matches, axis=0, num_outputs=self._num_images, squeeze_axis=True)
        # cls_targets (B, N) -> B * (N,)
        cls_targets = _split(cls_targets, axis=0, num_outputs=self._num_images, squeeze_axis=True)

        mask_targets = []
        mask_masks = []
        for roi, gt_mask, match, cls_target in zip(rois, gt_masks, matches, cls_targets):
            # batch id = match
            padded_rois = F.concat(match.reshape((-1, 1)), roi, dim=-1)
            # pooled_mask (N, 1, MS, MS) -> (N, MS, MS)
            pooled_mask = F.contrib.ROIAlign(gt_mask, padded_rois, self._mask_size, 1.0, sample_ratio=2)
            pooled_mask = pooled_mask.reshape((-3, 0, 0)) > 0.5
            # duplicate to C * (N, MS, MS)
            mask_target = [F.zeros_like(pooled_mask)]
            mask_mask = [F.zeros_like(pooled_mask)]
            for cid in range(1, self._num_classes):
                # boolean array (N,) -> (N, 1, 1)
                same_cid = (cls_target == cid).reshape((-1, 1, 1))
                # keep orig targets
                mask_target.append(pooled_mask)
                # but mask out the one not belong to this class [N, MS, MS]
                mask_mask.append(F.broadcast_mul(F.ones_like(pooled_mask), same_cid))
            # (C, N, MS, MS) -> (N, C, MS, MS)
            mask_targets.append(F.stack(*mask_target, axis=0).transpose((1, 0, 2, 3)))
            mask_masks.append(F.stack(*mask_mask, axis=0).transpose((1, 0, 2, 3)))

        # B * (N, C, MS, MS) -> (B, N, C, MS, MS)
        mask_targets = F.stack(*mask_targets, axis=0)
        mask_masks = F.stack(*mask_masks, axis=0)
        return mask_targets, mask_masks
