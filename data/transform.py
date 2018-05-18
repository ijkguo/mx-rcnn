import mxnet as mx

from data.image import imdecode, resize, transform
from net.config import *


def load_test(filename, short=SHORT_SIDE, max_size=LONG_SIDE,
              mean=PIXEL_MEANS, std=PIXEL_STDS):
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
    def __init__(self, short=SHORT_SIDE, max_size=LONG_SIDE, mean=PIXEL_MEANS, std=PIXEL_STDS):
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
