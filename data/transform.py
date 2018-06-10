import mxnet as mx

from data.image import imdecode, random_flip, resize, transform
from data.np_bbox import bbox_flip
from data.anchor import RPNAnchorGenerator, RPNTargetGenerator


def load_test(filename, short, max_size, mean, std):
    # read and transform image
    im_orig = imdecode(filename)
    im, im_scale = resize(im_orig, short, max_size)
    height, width = im.shape[:2]
    im_info = mx.nd.array([height, width, im_scale])

    # transform into tensor and normalize
    im_tensor = transform(im, mean, std)

    # for 1-batch inference purpose, cannot use batchify (or nd.stack) to expand dims
    im_tensor = im_tensor.expand_dims(0)
    im_info = im_info.expand_dims(0)

    return im_tensor, im_info, im_orig


class RCNNDefaultValTransform(object):
    def __init__(self, short, max_size, mean, std):
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std

    def __call__(self, src, label):
        # resize image
        im, im_scale = resize(src, self._short, self._max_size)
        height, width = im.shape[:2]
        im_info = mx.nd.array([height, width, im_scale])

        # transform into tensor and normalize
        im_tensor = transform(im, self._mean, self._std)
        return im_tensor, im_info, label


class RCNNDefaultTrainTransform(object):
    def __init__(self, short, max_size, mean, std, ac, rag: RPNAnchorGenerator, rtg: RPNTargetGenerator):
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std
        self._ac = ac
        self._rag = rag
        self._rtg = rtg

    def __call__(self, src, label):
        # random flip image
        im, flip_x = random_flip(src, px=0.5)

        # resize image
        im, im_scale = resize(im, self._short, self._max_size)
        im_height, im_width = im.shape[:2]
        im_info = mx.nd.array([im_height, im_width, im_scale])

        # transform into tensor and normalize
        im_tensor = transform(im, self._mean, self._std)

        # label is (np.array) gt_boxes [n, 6] (x1, y1, x2, y2, cls, difficult)
        # proposal target input (x1, y1, x2, y2)
        # important: need to copy this np.array (numpy will keep this array)
        gt_bboxes = label[:, :5].copy()

        # resize bbox
        gt_bboxes[:, :4] *= im_scale

        # add 1 to gt_bboxes for bg class
        gt_bboxes[:, 4] += 1

        # random flip bbox
        gt_bboxes = bbox_flip(gt_bboxes, im_width, flip_x)

        # convert to ndarray
        gt_bboxes = mx.nd.array(gt_bboxes, ctx=im_tensor.context)

        # compute anchor shape and generate anchors
        feat_height, feat_width = self._ac(im_height), self._ac(im_width)
        anchors = self._rag.forward(feat_height, feat_width).as_in_context(im_tensor.context)

        # assign anchors
        boxes = gt_bboxes[:, :4].expand_dims(axis=0)
        cls_target, box_target, box_mask = self._rtg.forward(boxes, anchors, im_width, im_height)

        cls_target = cls_target.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1))
        box_target = box_target.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1))
        box_mask = box_mask.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1))

        # need different repr
        cls_target = cls_target.reshape((-3, 0)).expand_dims(0)
        cls_mask = mx.nd.where(cls_target >= 0, mx.nd.ones_like(cls_target), mx.nd.zeros_like(cls_target))
        cls_target = mx.nd.where(cls_target >= 0, cls_target, mx.nd.zeros_like(cls_target))

        return im_tensor, im_info, gt_bboxes, cls_target, cls_mask, box_target, box_mask
