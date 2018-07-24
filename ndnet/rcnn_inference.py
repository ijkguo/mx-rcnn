from mxnet import gluon
from ndnet.bbox import BBoxClipper, BBoxCornerToCenter
from ndnet.coder import NormalizedBoxCenterDecoder


class RCNNDetector(gluon.HybridBlock):
    def __init__(self, clip, rcnn_bbox_stds, rcnn_num_classes, rcnn_nms_thresh, rcnn_nms_topk):
        super(RCNNDetector, self).__init__()
        self._bbox_corner2center = BBoxCornerToCenter()
        self._bbox_decoder = NormalizedBoxCenterDecoder(stds=rcnn_bbox_stds, clip=clip)
        self._bbox_clip = BBoxClipper()
        self._rcnn_num_classes = rcnn_num_classes
        self._rcnn_nms_thresh = rcnn_nms_thresh
        self._rcnn_nms_topk = rcnn_nms_topk

    def hybrid_forward(self, F, rois, cls, reg, im_info, **kwargs):
        # rois [N, 4], reg [C, N, 4] -> bboxes [C, N, 4]
        bboxes = self._bbox_decoder(reg, self._bbox_corner2center(rois))
        # im_info [2] -> [1, 2], clipped [C, N, 4]
        bboxes = self._bbox_clip(bboxes, im_info.expand_dims(0))

        # cls [C, N, 1] -> det [C, N, 5]
        det = F.concat(cls, bboxes, dim=-1)
        det = F.contrib.box_nms(det, valid_thresh=0.0001, overlap_thresh=self._rcnn_nms_thresh, topk=self._rcnn_nms_topk,
                                id_index=-1, score_index=0, coord_start=1, force_suppress=True)
        scores = F.slice_axis(det, axis=-1, begin=0, end=1)
        bboxes = F.slice_axis(det, axis=-1, begin=1, end=5)
        return scores, bboxes


class RCNNBatchDetector(gluon.HybridBlock):
    def __init__(self, batch_images, clip, rcnn_bbox_stds, rcnn_num_classes, rcnn_nms_thresh, rcnn_nms_topk):
        super(RCNNBatchDetector, self).__init__()
        self._batch_images = batch_images
        self._detector = RCNNDetector(clip, rcnn_bbox_stds, rcnn_num_classes, rcnn_nms_thresh, rcnn_nms_topk)

    def hybrid_forward(self, F, ids, rois, cls, reg, im_info, **kwargs):
        # ids [B, C, N, 1] rois [B, C, N, 4], cls [B, C, N, 1], reg [B, C, N, 1], im_info [B, 2]
        # ret_ids [B, C, topk, 1], ret_scores [B, C, topk, 1], ret_bboxes [B, C, topk, 4]
        ret_ids = []
        ret_scores = []
        ret_bboxes = []
        for i in range(self._batch_images):
            b_rois = F.squeeze(F.slice_axis(rois, axis=0, begin=i, end=i+1), axis=0)
            b_cls = F.squeeze(F.slice_axis(cls, axis=0, begin=i, end=i+1), axis=0)
            b_reg = F.squeeze(F.slice_axis(reg, axis=0, begin=i, end=i+1), axis=0)
            b_im_info = F.squeeze(F.slice_axis(im_info, axis=0, begin=i, end=i+1), axis=0)
            scores, bboxes = self._detector(b_rois, b_cls, b_reg, b_im_info)
            b_ids = F.where(scores < 0, F.ones_like(ids) * -1, ids)
            ret_ids.append(b_ids.reshape((-1, 1)))
            ret_scores.append(scores.reshape((-1, 1)))
            ret_bboxes.append(bboxes.reshape((-1, 4)))
        ret_ids = F.stack(*ret_ids, axis=0)
        ret_scores = F.stack(*ret_scores, axis=0)
        ret_bboxes = F.stack(*ret_bboxes, axis=0)
        return ret_ids, ret_scores, ret_bboxes
