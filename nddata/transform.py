import mxnet as mx
from mxnet import gluon

from nddata.image import imdecode, random_flip, resize, transform
from ndnet.rpn_target import RPNTargetGenerator
from symdata.bbox import bbox_flip
from symdata.mask import polys_flip, polys_to_mask


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
    anchors = anchors[:feat_height, :feat_width, :].reshape((-1, 4))

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
        anchors = self._anchors[:feat_height, :feat_width, :].reshape((-1, 4))
        anchors = anchors.as_in_context(src.context)
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
        gt_bboxes = mx.nd.array(gt_bboxes, ctx=src.context)

        # compute real anchor shape and slice anchors to this shape
        feat_height, feat_width = self._asf(im_height, im_width)
        anchors = self._anchors[:feat_height, :feat_width, :].reshape((-1, 4))
        anchors = anchors.as_in_context(src.context)

        # assign anchors
        boxes = gt_bboxes[:, :4]
        cls_target, box_target, box_mask = self._rtg.forward(boxes, anchors, im_width, im_height)
        return im_tensor, anchors, im_info, gt_bboxes, cls_target, box_target, box_mask


class MaskDefaultValTransform(object):
    def __init__(self, short, max_size, mean, std, anchors, asf):
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std
        self._anchors = anchors
        self._asf = asf

    def __call__(self, src, label, mask):
        # resize image
        im, im_scale = resize(src, self._short, self._max_size)
        height, width = im.shape[:2]
        im_info = mx.nd.array([height, width, im_scale], ctx=src.context)

        # transform into tensor and normalize
        im_tensor = transform(im, self._mean, self._std)
        label = mx.nd.array(label, ctx=src.context)

        # compute real anchor shape and slice anchors to this shape
        feat_height, feat_width = self._asf(height, width)
        anchors = self._anchors[:feat_height, :feat_width, :].reshape((-1, 4))
        anchors = anchors.as_in_context(src.context)
        return im_tensor, anchors, im_info, label


class MaskDefaultTrainTransform(object):
    def __init__(self, short, max_size, mean, std, anchors, asf, rtg: RPNTargetGenerator):
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std
        self._anchors = anchors
        self._asf = asf
        self._rtg = rtg

    def __call__(self, src, label, segm):
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
        gt_bboxes = mx.nd.array(gt_bboxes, ctx=src.context)

        # gt_masks (n, im_height, im_width) of uint8 -> np.float32 (cannot take uint8)
        gt_masks = []
        for polys in segm:
            # resize poly
            polys = [poly * im_scale for poly in polys]
            # random flip
            polys = polys_flip(polys, im_width, flip_x)
            # poly to mask
            mask = mx.nd.array(polys_to_mask(polys, im_height, im_width), ctx=src.context)
            gt_masks.append(mask)
        # n * (im_height, im_width) -> (n, im_height, im_width)
        gt_masks = mx.nd.stack(*gt_masks, axis=0)

        # compute real anchor shape and slice anchors to this shape
        feat_height, feat_width = self._asf(im_height, im_width)
        anchors = self._anchors[:feat_height, :feat_width, :].reshape((-1, 4))
        anchors = anchors.as_in_context(src.context)

        # assign anchors
        boxes = gt_bboxes[:, :4]
        cls_target, box_target, box_mask = self._rtg.forward(boxes, anchors, im_width, im_height)
        return im_tensor, anchors, im_info, gt_bboxes, gt_masks, cls_target, box_target, box_mask
