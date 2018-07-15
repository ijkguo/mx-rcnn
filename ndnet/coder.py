from mxnet import gluon
from ndnet.bbox import BBoxCornerToCenter
from gluoncv.nn.coder import SigmoidClassEncoder

__all__ = ['SigmoidClassEncoder', 'MultiClassEncoder',
           'NormalizedBoxCenterEncoder', 'NormalizedPerClassBoxCenterEncoder', 'NormalizedBoxCenterDecoder']


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


class NormalizedSimpleBoxCenterEncoder(gluon.HybridBlock):
    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.)):
        super(NormalizedSimpleBoxCenterEncoder, self).__init__()
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        self._stds = stds
        self._means = means

    def hybrid_forward(self, F, ref_boxes, boxes):
        # ref_boxes: (..., 4) gx, gy, gw, gh
        # boxes: (..., 4) ax, ay, aw, ah
        g = F.split(ref_boxes, axis=-1, num_outputs=4)
        a = F.split(boxes, axis=-1, num_outputs=4)
        tx = (F.broadcast_div(F.broadcast_minus(g[0], a[0]), a[2]) - self._means[0]) / self._stds[0]
        ty = (F.broadcast_div(F.broadcast_minus(g[1], a[1]), a[3]) - self._means[1]) / self._stds[1]
        tw = (F.log(F.broadcast_div(g[2], a[2])) - self._means[2]) / self._stds[2]
        th = (F.log(F.broadcast_div(g[3], a[3])) - self._means[3]) / self._stds[3]
        return F.concat(tx, ty, tw, th, dim=-1)


class NormalizedBoxCenterDecoder(gluon.HybridBlock):
    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.), clip=4.42):
        super(NormalizedBoxCenterDecoder, self).__init__()
        assert len(stds) == 4, "Box Decoder requires 4 std values."
        self._stds = stds
        self._means = means
        self._clip = clip

    def hybrid_forward(self, F, targets, boxes):
        # targets: (..., 4) tx, ty, tw, th
        # boxes: (..., 4) ax, ay, aw, th
        p = F.split(targets, axis=-1, num_outputs=4)
        a = F.split(boxes, axis=-1, num_outputs=4)
        ox = F.broadcast_add(F.broadcast_mul(p[0] * self._stds[0] + self._means[0], a[2]), a[0])
        oy = F.broadcast_add(F.broadcast_mul(p[1] * self._stds[1] + self._means[1], a[3]), a[1])
        ow = F.broadcast_mul(F.minimum(F.exp(p[2] * self._stds[2] + self._means[2]), self._clip), a[2]) / 2
        oh = F.broadcast_mul(F.minimum(F.exp(p[3] * self._stds[3] + self._means[3]), self._clip), a[3]) / 2
        return F.concat(ox - ow, oy - oh, ox + ow, oy + oh, dim=-1)


class NormalizedBoxCenterEncoder(gluon.HybridBlock):
    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.)):
        super(NormalizedBoxCenterEncoder, self).__init__()
        with self.name_scope():
            self.corner_to_center = BBoxCornerToCenter()
            self.box_encoder = NormalizedSimpleBoxCenterEncoder(stds, means)

    def hybrid_forward(self, F, samples, matches, anchors, refs):
        # refs [B, M, 4], anchors [B, N, 4], samples [B, N], matches [B, N]
        # refs [B, M, 4] -> reshape [B, 1, M, 4] -> repeat [B, N, M, 4]
        ref_boxes = F.repeat(refs.reshape((0, 1, -1, 4)), axis=1, repeats=matches.shape[1])
        # refs [B, N, M, 4] -> 4 * [B, N, M]
        ref_boxes = F.split(ref_boxes, axis=-1, num_outputs=4, squeeze_axis=True)
        # refs 4 * [B, N, M] -> pick from matches [B, N, 1] -> concat to [B, N, 4]
        ref_boxes = F.concat(*[F.pick(ref_boxes[i], matches, axis=2).reshape((0, -1, 1)) \
            for i in range(4)], dim=2)
        # transform based on x, y, w, h
        # g [B, N, 4], a [B, N, 4] -> codecs [B, N, 4]
        g = self.corner_to_center(ref_boxes)
        a = self.corner_to_center(anchors)
        codecs = self.box_encoder(g, a)
        # samples [B, N] -> [B, N, 1] -> [B, N, 4] -> boolean
        temp = F.tile(samples.reshape((0, -1, 1)), reps=(1, 1, 4)) > 0.5
        # fill targets and masks [B, N, 4]
        targets = F.where(temp, codecs, F.zeros_like(codecs))
        masks = F.where(temp, F.ones_like(temp), F.zeros_like(temp))
        return targets, masks


class NormalizedPerClassBoxCenterEncoder(gluon.HybridBlock):
    def __init__(self, num_class, num_sample, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.)):
        super(NormalizedPerClassBoxCenterEncoder, self).__init__()
        self._num_class = num_class
        self._num_sample = num_sample
        with self.name_scope():
            self.corner_to_center = BBoxCornerToCenter()
            self.box_encoder = NormalizedSimpleBoxCenterEncoder(stds, means)

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
        # g [B, N, 4] a [B, N, 4] -> codecs [B, N, 4]
        g = self.corner_to_center(ref_boxes)
        a = self.corner_to_center(anchors)
        codecs = self.box_encoder(g, a)
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
