import mxnet as mx
from mxnet import gluon
from data.bbox import bbox_center2corner, bbox_corner2center, bbox_decode, bbox_clip
from data.np_anchor import AnchorGenerator


class Proposal(gluon.HybridBlock):
    def __init__(self, anchor_scales, anchor_ratios, rpn_feature_stride,
                 rpn_pre_topk, rpn_post_topk, rpn_nms_thresh, rpn_min_size,
                 alloc_size=(128, 128), output_score=False, **kwargs):
        super(Proposal, self).__init__(**kwargs)
        ag = AnchorGenerator(rpn_feature_stride, anchor_scales, anchor_ratios)
        self._anchors = mx.nd.array(
            ag.generate(*alloc_size).reshape((1, 1, alloc_size[0], alloc_size[1], -1)))
        self._rpn_pre_topk = rpn_pre_topk
        self._rpn_post_topk = rpn_post_topk
        self._rpn_nms_thresh = rpn_nms_thresh
        self._rpn_min_size = rpn_min_size
        self._output_score = output_score

    def _generate_anchor(self, x):
        return mx.nd.slice_like(self._anchors.as_in_context(x.context), x, axes=(2, 3)).reshape(1, -1, 4)

    def hybrid_forward(self, F, cls, reg, im_info, **kwargs):
        # nd proposal
        anchors = self._generate_anchor(reg)

        score = cls.transpose((0, 2, 3, 1)).reshape((0, -1, 1))
        bbox_reg = reg.transpose((0, 2, 3, 1)).reshape((0, -1, 4))

        boxes = bbox_corner2center(anchors)
        boxes = bbox_decode(bbox_reg, boxes, stds=(1.0, 1.0, 1.0, 1.0))
        boxes = bbox_center2corner(boxes)

        # clip to image boundary
        for ib, [im_height, im_width, im_scale] in enumerate(im_info.asnumpy()):
            boxes[ib] = bbox_clip(boxes[ib], im_height, im_width)

        # remove min_size
        for ib, [im_height, im_width, im_scale] in enumerate(im_info.asnumpy()):
            _, _, width, height = bbox_corner2center(boxes[ib], split=True)
            min_size = self._rpn_min_size * im_scale
            invalid = (width < min_size) + (height < min_size)
            score[ib] = F.where(invalid, F.zeros_like(invalid), score[ib])
            invalid = F.repeat(invalid, axis=-1, repeats=4)
            boxes[ib] = F.where(invalid, F.ones_like(invalid) * -1, boxes[ib])

        # nms
        # Non-maximum suppression
        pre = F.concat(score, boxes, dim=-1)
        tmp = F.contrib.box_nms(pre, self._rpn_nms_thresh, self._rpn_pre_topk, coord_start=1,
                                score_index=0, id_index=-1, force_suppress=True)

        # slice post_nms number of boxes
        result = F.slice_axis(tmp, axis=1, begin=0, end=self._rpn_post_topk)
        rpn_scores = F.slice_axis(result, axis=-1, begin=0, end=1)
        rpn_bbox = F.slice_axis(result, axis=-1, begin=1, end=None)

        if self._output_score:
            return rpn_bbox, rpn_scores
        return rpn_bbox
