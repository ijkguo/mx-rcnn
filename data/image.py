import mxnet as mx


def imdecode(image_name):
    """Return NDArray [height, width, channel]"""
    import os
    assert os.path.exists(image_name), image_name + ' not found'
    with open(image_name, 'rb') as f:
        buf = f.read()
    im = mx.image.imdecode(buf)
    return im


def resize(im, short, max_size):
    """Return resized image (NDArray) and scale (float)"""
    im_shape = im.shape
    im_size_min = min(im_shape[0:2])
    im_size_max = max(im_shape[0:2])
    im_scale = float(short) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
        short = int(im_size_min * im_scale)
    im = mx.image.resize_short(im, short)
    return im, im_scale


def transform(im, means, stds):
    """transform from RGB image into (C, H, W) tensor with RGB means and stds"""
    im_tensor = im.transpose((2, 0, 1)).astype("float32")
    im_tensor = mx.nd.image.normalize(im_tensor, means, stds)
    return im_tensor


def transform_inverse(im_tensor, means, stds):
    """transform from (C, H, W) tensor to RGB image"""
    means = mx.nd.array(means).reshape((1, 3))
    stds = mx.nd.array(stds).reshape((1, 3))
    im = im_tensor.transpose((1, 2, 0)) * stds + means
    return im.astype(int)
