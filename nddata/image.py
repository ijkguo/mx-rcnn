import mxnet as mx
import numpy as np


def imdecode(image_name):
    """Return NDArray [height, width, channel]"""
    import os
    assert os.path.exists(image_name), image_name + ' not found'
    with open(image_name, 'rb') as f:
        buf = f.read()
    im = mx.image.imdecode(buf)
    return im


def random_flip(src, px=0):
    """Flip x axis of image [height, width, channel] with prob px"""
    flip_x = np.random.choice([False, True], p=[1 - px, px])
    if flip_x:
        src = mx.nd.flip(src, axis=1)
    return src, flip_x


def resize(im, short, max_size):
    """Return resized image (NDArray) and scale (float)"""
    h, w = im.shape[:2]
    im_size_min = min(h, w)
    im_size_max = max(h, w)
    im_scale = float(short) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    new_h, new_w = int(round(h * im_scale)), int(round(w * im_scale))
    im = mx.image.imresize(im, new_w, new_h, interp=1)
    return im, im_scale


def transform(im, means, stds):
    """transform from RGB image into (C, H, W) tensor with RGB means and stds"""
    im_tensor = im.transpose((2, 0, 1)).astype("float32")
    im_tensor = mx.nd.image.normalize(im_tensor, means, stds)
    return im_tensor


def transform_inverse(im_tensor, means, stds):
    """transform from (C, H, W) tensor to RGB image"""
    means = mx.nd.array(means, ctx=im_tensor.context).reshape((1, 3))
    stds = mx.nd.array(stds, ctx=im_tensor.context).reshape((1, 3))
    im = im_tensor.transpose((1, 2, 0)) * stds + means
    return im.astype(int)
