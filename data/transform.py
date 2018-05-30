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

        # label is (np.array) gt_boxes [n, 6] (x1, y1, x2, y2, cls, difficult)
        # proposal target input (x1, y1, x2, y2)
        # important: need to copy this np.array (numpy will keep this array)
        gt_bboxes = label[:, :5].copy()

        # resize bbox
        gt_bboxes[:, :4] *= im_scale

        # random flip image and bbox
        im, flip_x = random_flip(im, px=0.5)
        # invalid value in bbox_transform if this wrong (no overlap), note index 0 and 2
        if flip_x:
            gt_bboxes[:, 0] = im_width - 1 - gt_bboxes[:, 2]
            gt_bboxes[:, 2] = im_width - 1 - gt_bboxes[:, 0]

        # compute anchor shape and generate anchors
        feat_height, feat_width = self._ac(im_height), self._ac(im_width)
        anchors = self._ag.generate(feat_height, feat_width)

        # assign anchors
        rpn_label, bbox_target, bbox_weight = self._asp.assign(anchors, gt_bboxes, im_height, im_width)

        rpn_label = rpn_label.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1)).flatten()
        bbox_target = bbox_target.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1))
        bbox_weight = bbox_weight.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1))

        # convert to ndarray
        gt_bboxes = mx.nd.array(gt_bboxes)
        rpn_label = mx.nd.array(rpn_label)
        bbox_target = mx.nd.array(bbox_target)
        bbox_weight = mx.nd.array(bbox_weight)
        return im_tensor, im_info, gt_bboxes, rpn_label, bbox_target, bbox_weight
