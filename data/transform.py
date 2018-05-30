import mxnet as mx

from data.image import imdecode, random_flip, resize, transform
from data.np_anchor import AnchorGenerator, AnchorSampler


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


def generate_batch(im_tensor, im_info):
    """return batch"""
    data = [im_tensor, im_info]
    data_shapes = [('data', im_tensor.shape), ('im_info', im_info.shape)]
    data_batch = mx.io.DataBatch(data=data, label=None, provide_data=data_shapes, provide_label=None)
    return data_batch


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
    def __init__(self, short, max_size, mean, std, ac, ag: AnchorGenerator, asp: AnchorSampler):
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std
        self._ac = ac
        self._ag = ag
        self._asp = asp

    def __call__(self, src, label):
        # resize image
        im, im_scale = resize(src, self._short, self._max_size)
        im_height, im_width = im.shape[:2]
        im_info = mx.nd.array([im_height, im_width, im_scale])

        # transform into tensor and normalize
        im_tensor = transform(im, self._mean, self._std)

        # label is gt_boxes [n, 6] (x1, y1, x2, y2, cls, difficult)
        # proposal target input (x1, y1, x2, y2)
        gt_bboxes = label.slice_axis(axis=-1, begin=0, end=5)

        # resize bbox
        gt_bboxes[:, :4] *= im_scale

        # random flip image and bbox
        im, flip_x = random_flip(im, px=0.5)
        if flip_x:
            gt_bboxes[:, 0] = im_height + 1 - gt_bboxes[:, 0]
            gt_bboxes[:, 2] = im_height + 1 - gt_bboxes[:, 2]

        # compute anchor shape and generate anchors
        feat_height, feat_width = self._ac(im_height, im_width)
        anchors = self._ag.generate(feat_height, feat_width)

        # assign anchors
        rpn_label, bbox_target, bbox_weight = self._asp.assign(anchors, gt_bboxes.asnumpy(), im_height, im_width)
        return im_tensor, im_info, gt_bboxes, rpn_label, bbox_target, bbox_weight
