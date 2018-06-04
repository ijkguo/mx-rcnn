import mxnet as mx
from gluoncv import data as gdata

from data.image import imdecode, random_flip, resize, transform
from data.np_bbox import bbox_flip
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
        # random flip image
        im, flip_x = random_flip(src, px=0.5)

        # resize image
        im, im_scale = resize(im, self._short, self._max_size)
        im_height, im_width = im.shape[:2]
        im_info = mx.nd.array([im_height, im_width, im_scale])

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

        # transform into tensor and normalize
        im_tensor = transform(im, self._mean, self._std)

        # compute anchor shape and generate anchors
        feat_height, feat_width = self._ac(im_height), self._ac(im_width)
        anchors = self._ag.generate(feat_height, feat_width)

        # assign anchors
        rpn_label, bbox_target, bbox_weight = self._asp.assign(anchors, gt_bboxes, im_height, im_width)

        rpn_label = rpn_label.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1))
        bbox_target = bbox_target.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1))
        bbox_weight = bbox_weight.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1))

        # convert to ndarray
        gt_bboxes = mx.nd.array(gt_bboxes)
        rpn_label = mx.nd.array(rpn_label)
        bbox_target = mx.nd.array(bbox_target)
        bbox_weight = mx.nd.array(bbox_weight)
        return im_tensor, im_info, gt_bboxes, rpn_label, bbox_target, bbox_weight


class RCNNGluonTrainTransform(RCNNDefaultTrainTransform):
    def __init__(self, short, max_size, mean, std, ac, ag: AnchorGenerator, asp: AnchorSampler):
        super(RCNNGluonTrainTransform, self).__init__(short, max_size, mean, std, ac, ag, asp)

    def __call__(self, src, label):
        im_tensor, im_info, gt_bboxes, rpn_label, bbox_target, bbox_weight = \
            super(RCNNGluonTrainTransform, self).__call__(src, label)

        # need different repr
        rpn_label = rpn_label.reshape((-3, 0)).expand_dims(0)
        rpn_weight = mx.nd.where(rpn_label >= 0, mx.nd.ones_like(rpn_label), mx.nd.zeros_like(rpn_label))
        rpn_label = mx.nd.where(rpn_label >= 0, rpn_label, mx.nd.zeros_like(rpn_label))

        return im_tensor, im_info, gt_bboxes, rpn_label, rpn_weight, bbox_target, bbox_weight


class AnchorIter(mx.io.DataIter):
    def __init__(self, dataset, batch_size, shuffle, last_batch, num_workers):
        super(AnchorIter, self).__init__(batch_size)
        self._loader = gdata.DetectionDataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                 last_batch=last_batch, num_workers=num_workers)
        self._iter = iter(self._loader)

        # decide data and label names
        self._data_name = ['data', 'im_info', 'gt_boxes']
        self._label_name = ['label', 'bbox_target', 'bbox_weight']
        self._data = None
        self._label = None

        # get first batch to fill in provide_data and provide_label
        self.next()
        self.reset()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self._data_name, self._data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self._label_name, self._label)]

    def reset(self):
        self._iter = iter(self._loader)

    def next(self):
        data, im_info, gt_boxes, label, bbox_target, bbox_weight = next(self._iter)
        self._data = [data, im_info, gt_boxes]
        self._label = [label, bbox_target, bbox_weight]
        return mx.io.DataBatch(data=self._data, label=self._label, pad=0, index=None,
                               provide_data=self.provide_data, provide_label=self.provide_label)
