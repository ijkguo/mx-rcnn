import mxnet as mx

from data.image import imdecode, resize, transform


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
