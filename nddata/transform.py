import mxnet as mx
from mxnet import gluon

from nddata.image import imdecode, random_flip, resize, transform
from ndnet.rpn_target import RPNTargetGenerator
from symdata.bbox import bbox_flip


def load_test(filename, short, max_size, mean, std, anchors, asf):
    # read and transform image
    im_orig = imdecode(filename)
    im, im_scale = resize(im_orig, short, max_size)
    height, width = im.shape[:2]
    im_info = mx.nd.array([height, width, im_scale])

    # transform into tensor and normalize
    im_tensor = transform(im, mean, std)

    # compute real anchor shape and slice anchors to this shape
    feat_height, feat_width = asf(height, width)
    anchors = anchors[:feat_height, :feat_width, :]

    # for 1-batch inference purpose, cannot use batchify (or nd.stack) to expand dims
    im_tensor = im_tensor.expand_dims(0)
    anchors = anchors.expand_dims(0)
    im_info = im_info.expand_dims(0)

    return im_tensor, anchors, im_info, im_orig


def batchify_pad(tensors_list):
    batch_size = len(tensors_list)
    num_tensor = len(tensors_list[0])
    all_tensor_list = []

    if batch_size == 1:
        for i in range(num_tensor):
            all_tensor_list.append(tensors_list[0][i].expand_dims(0))
        return all_tensor_list

    for i in range(num_tensor):
        ndim = len(tensors_list[0][i].shape)
        if ndim > 3:
            raise Exception('Sorry, unimplemented.')

        dimensions = [batch_size]
        for dim in range(ndim):
            dimensions.append(max(tensors_list[j][i].shape[dim] for j in range(batch_size)))

        all_tensor = mx.nd.zeros(tuple(dimensions), mx.Context('cpu_shared', 0))
        if ndim == 1:
            for j in range(batch_size):
                all_tensor[j, :tensors_list[j][i].shape[0]] = tensors_list[j][i]
        elif ndim == 2:
            for j in range(batch_size):
                all_tensor[j, :tensors_list[j][i].shape[0], :tensors_list[j][i].shape[1]] = tensors_list[j][i]
        elif ndim == 3:
            for j in range(batch_size):
                all_tensor[j, :tensors_list[j][i].shape[0], :tensors_list[j][i].shape[1],
                :tensors_list[j][i].shape[2]] = tensors_list[j][i]

        all_tensor_list.append(all_tensor)
    return all_tensor_list


def split_pad(batch, ctx_list):
    return [gluon.utils.split_and_load(data, ctx_list) for data in batch]


def batchify_append(tensors_list):
    batch_size = len(tensors_list)
    num_tensor = len(tensors_list[0])
    all_tensor_list = []

    if batch_size == 1:
        for i in range(num_tensor):
            all_tensor_list.append([tensors_list[0][i].expand_dims(0)])
        return all_tensor_list

    for i in range(num_tensor):
        batches = []
        for j in range(batch_size):
            batches.append(tensors_list[j][i].expand_dims(0))
        all_tensor_list.append(batches)
    return all_tensor_list


def split_append(batch, ctx_list):
    return [[x.as_in_context(c) for x, c in zip(data, ctx_list)] for data in batch]


class RCNNDefaultValTransform(object):
    def __init__(self, short, max_size, mean, std, anchors, asf):
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std
        self._anchors = anchors
        self._asf = asf

    def __call__(self, src, label):
        # resize image
        im, im_scale = resize(src, self._short, self._max_size)
        height, width = im.shape[:2]
        im_info = mx.nd.array([height, width, im_scale], ctx=src.context)

        # transform into tensor and normalize
        im_tensor = transform(im, self._mean, self._std)
        label = mx.nd.array(label, ctx=src.context)

        # compute real anchor shape and slice anchors to this shape
        feat_height, feat_width = self._asf(height, width)
        anchors = self._anchors[:feat_height, :feat_width, :].as_in_context(src.context)
        return im_tensor, anchors, im_info, label


class RCNNDefaultTrainTransform(object):
    def __init__(self, short, max_size, mean, std, anchors, asf, rtg: RPNTargetGenerator):
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std
        self._anchors = anchors
        self._asf = asf
        self._rtg = rtg

    def __call__(self, src, label):
        # random flip image
        im, flip_x = random_flip(src, px=0.5)

        # resize image
        im, im_scale = resize(im, self._short, self._max_size)
        im_height, im_width = im.shape[:2]
        im_info = mx.nd.array([im_height, im_width, im_scale], ctx=src.context)

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

        # compute real anchor shape and slice anchors to this shape
        feat_height, feat_width = self._asf(im_height, im_width)
        anchors = self._anchors[:feat_height, :feat_width, :].as_in_context(im_tensor.context)

        # assign anchors
        boxes = gt_bboxes[:, :4]
        cls_target, box_target, box_mask = self._rtg.forward(boxes, anchors.reshape(-1, 4), im_width, im_height)

        cls_target = cls_target.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1))
        box_target = box_target.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1))
        box_mask = box_mask.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1))

        return im_tensor, anchors, im_info, gt_bboxes, cls_target, box_target, box_mask
