from mxnet import gluon
from ndnet.bbox import BBoxCornerToCenter
from gluoncv.nn.coder import NormalizedBoxCenterDecoder

__all__ = ['MultiClassEncoder', 'NormalizedPerClassBoxCenterEncoder', 'NormalizedBoxCenterDecoder']


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
