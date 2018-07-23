from mxnet import gluon
from ndnet.bbox import BBoxCornerToCenter, BBoxClipper
from ndnet.coder import NormalizedBoxCenterDecoder


class Proposal(gluon.HybridBlock):
    def __init__(self, rpn_pre_topk, rpn_post_topk, rpn_nms_thresh, rpn_min_size,
                 output_score=False, **kwargs):
        super(Proposal, self).__init__(**kwargs)
        self._rpn_pre_topk = rpn_pre_topk
        self._rpn_post_topk = rpn_post_topk
        self._rpn_nms_thresh = rpn_nms_thresh
        self._rpn_min_size = rpn_min_size
        self._output_score = output_score

        with self.name_scope():
            self._bbox_corner2center = BBoxCornerToCenter()
            self._bbox_decode = NormalizedBoxCenterDecoder(stds=(1.0, 1.0, 1.0, 1.0))
            self._bbox_clip = BBoxClipper()
            self._bbox_corner2center_split = BBoxCornerToCenter(split=True)

    def hybrid_forward(self, F, cls, reg, anchors, im_info, **kwargs):
        # anchors [B, H, W, N*4] -> split 1st dim [B, 1, H, W, N*4] -> slice 2, 3 dim -> reshape [B, H * W * N, 4]
        anchors = F.slice_like(anchors.reshape((-4, -1, 1, -2)), reg, axes=(2, 3)).reshape((0, -1, 4))

        # reshape input [B, N*4, H, W] -> [B, H * W * N, 4]
        score = cls.transpose((0, 2, 3, 1)).reshape((0, -1, 1))
        bbox_reg = reg.transpose((0, 2, 3, 1)).reshape((0, -1, 4))

        # decode bbox
        boxes = self._bbox_decode(bbox_reg, self._bbox_corner2center(anchors))

        # clip to image boundary
        boxes = self._bbox_clip(boxes, im_info.slice_axis(axis=-1, begin=0, end=2))

        # remove min_size
        x_ctr, y_ctr, width, height = self._bbox_corner2center_split(boxes)
        invalid = (width < self._rpn_min_size) + (height < self._rpn_min_size) \
                  + F.broadcast_greater_equal(x_ctr, im_info.slice_axis(axis=-1, begin=1, end=2)) \
                  + F.broadcast_greater_equal(y_ctr, im_info.slice_axis(axis=-1, begin=0, end=1))
        score = F.where(invalid, F.zeros_like(invalid), score)
        invalid = F.repeat(invalid, axis=-1, repeats=4)
        boxes = F.where(invalid, F.ones_like(invalid) * -1, boxes)

        # nms
        # Non-maximum suppression
        pre = F.concat(score, boxes, dim=-1)
        tmp = F.contrib.box_nms(pre, overlap_thresh=self._rpn_nms_thresh, topk=self._rpn_pre_topk,
                                coord_start=1, score_index=0, id_index=-1, force_suppress=True)

        # slice post_nms number of boxes
        result = F.slice_axis(tmp, axis=1, begin=0, end=self._rpn_post_topk)
        rpn_scores = F.slice_axis(result, axis=-1, begin=0, end=1)
        rpn_bbox = F.slice_axis(result, axis=-1, begin=1, end=None)

        if self._output_score:
            return rpn_bbox, rpn_scores
        return rpn_bbox
