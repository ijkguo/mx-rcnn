import argparse
import mxnet as mx
import numpy as np

from rcnn.config import config
from rcnn.symbol.symbol_resnet import get_resnet_test
from rcnn.utils.load_model import load_param
from rcnn.core.tester import Predictor


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

SHORT_SIDE = config.SCALES[0][0]
LONG_SIDE = config.SCALES[0][1]

PIXEL_MEANS = (123.68, 116.779, 103.939)
PIXEL_STDS = (1.0, 1.0, 1.0)

DATA_NAMES = ['data', 'im_info']
LABEL_NAMES = None
DATA_SHAPES = [('data', (1, 3, LONG_SIDE, SHORT_SIDE)), ('im_info', (1, 3))]
LABEL_SHAPES = None

CONF_THRESH = 0.7
NMS_THRESH = 0.3


def imdecode(image_name):
    """Return NDArray [height, width, channel]"""
    import os
    assert os.path.exists(image_name), image_name + ' not found'
    with open(image_name, 'rb') as f:
        buf = f.read()
    im = mx.image.imdecode(buf)
    return im


def resize(im, short=SHORT_SIDE, max_size=LONG_SIDE):
    """Return resized image (NDArray) and scale (float)"""
    im_shape = im.shape
    im_size_min = min(im_shape[0:2])
    im_size_max = max(im_shape[0:2])
    im_scale = float(short) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = mx.image.resize_short(im, short)
    return im, im_scale


def transform(im, means=PIXEL_MEANS, stds=PIXEL_STDS):
    """transform from RGB image into (C, H, W) tensor with RGB means and stds"""
    im_tensor = im.transpose((2, 0, 1)).astype("float32")
    im_tensor = mx.nd.image.normalize(im_tensor, means, stds)
    return im_tensor


def transform_inverse(im_tensor, means=PIXEL_MEANS, stds=PIXEL_STDS):
    """transform from (C, H, W) tensor to RGB image"""
    means = mx.nd.array(means).reshape((1, 3))
    stds = mx.nd.array(stds).reshape((1, 3))
    im = im_tensor.transpose((1, 2, 0)) * stds + means
    return im


def get_net(symbol, prefix, epoch, ctx):
    arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)

    # infer shape
    data_shape_dict = dict(DATA_SHAPES)
    arg_names, aux_names = symbol.list_arguments(), symbol.list_auxiliary_states()
    arg_shape, _, aux_shape = symbol.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(arg_names, arg_shape))
    aux_shape_dict = dict(zip(aux_names, aux_shape))

    # check shapes
    for k in symbol.list_arguments():
        if k in data_shape_dict or 'label' in k:
            continue
        assert k in arg_params, k + ' not initialized'
        assert arg_params[k].shape == arg_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
    for k in symbol.list_auxiliary_states():
        assert k in aux_params, k + ' not initialized'
        assert aux_params[k].shape == aux_shape_dict[k], \
            'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

    predictor = Predictor(symbol, DATA_NAMES, LABEL_NAMES, context=ctx,
                          provide_data=DATA_SHAPES, provide_label=LABEL_SHAPES,
                          arg_params=arg_params, aux_params=aux_params)
    return predictor


def generate_batch(im_tensor, im_info):
    """return batch"""
    # for 1-batch inference purpose, cannot use batchify (or nd.stack) to expand dims
    im_tensor = im_tensor.expand_dims(0)
    im_info = im_info.expand_dims(0)
    data = [im_tensor, im_info]
    data_shapes = [('data', im_tensor.shape), ('im_info', im_info.shape)]
    data_batch = mx.io.DataBatch(data=data, label=None, provide_data=data_shapes, provide_label=None)
    return data_batch


def pick_deltas(cls, deltas):
    delta0 = deltas.pick(4 * cls, axis=1, keepdims=True)
    delta1 = deltas.pick(4 * cls + 1, axis=1, keepdims=True)
    delta2 = deltas.pick(4 * cls + 2, axis=1, keepdims=True)
    delta3 = deltas.pick(4 * cls + 3, axis=1, keepdims=True)
    return mx.nd.concat(delta0, delta1, delta2, delta3, dim=-1)


def bbox_corner2center(x, split=False):
    xmin, ymin, xmax, ymax = x.split(axis=-1, num_outputs=4)
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    x = xmin + (width - 1) / 2
    y = ymin + (height - 1) / 2
    if not split:
        return mx.nd.concat(x, y, width, height, dim=-1)
    else:
        return x, y, width, height


def bbox_center2corner(x, split=False):
    x, y, w, h = x.split(axis=-1, num_outputs=4)
    hw = (w - 1) / 2
    hh = (h - 1) / 2
    xmin = x - hw
    ymin = y - hh
    xmax = x + hw
    ymax = y + hh
    if not split:
        return mx.nd.concat(xmin, ymin, xmax, ymax, dim=-1)
    else:
        return xmin, ymin, xmax, ymax


def bbox_decode(x, anchors, stds=(1.0, 1.0, 1.0, 1.0)):
    ax, ay, aw, ah = anchors.split(axis=-1, num_outputs=4)
    dx, dy, dw, dh = x.split(axis=-1, num_outputs=4)
    ox = dx * stds[0] * aw + ax
    oy = dy * stds[1] * ah + ay
    ow = mx.nd.exp(dw * stds[2]) * aw
    oh = mx.nd.exp(dh * stds[3]) * ah
    return mx.nd.concat(ox, oy, ow, oh, dim=-1)


def bbox_clip(x, height, width):
    xmin, ymin, xmax, ymax = x.split(axis=-1, num_outputs=4)
    xmin = xmin.clip(0, width)
    ymin = ymin.clip(0, height)
    xmax = xmax.clip(0, width)
    ymax = ymax.clip(0, height)
    return mx.nd.concat(xmin, ymin, xmax, ymax, dim=-1)


def vis_detection(im_array, detections, class_names):
    """visualize [cls, conf, x1, y1, x2, y2]"""
    import matplotlib.pyplot as plt
    import random
    plt.imshow(im_array)
    colors = [(random.random(), random.random(), random.random()) for _ in class_names]
    for [cls, conf, x1, y1, x2, y2] in detections:
        if cls > 0:
            cls = int(cls)
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, edgecolor=colors[cls], linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(x1, y1 - 2, '{:s} {:.3f}'.format(class_names[cls], conf),
                           bbox=dict(facecolor=colors[cls], alpha=0.5), fontsize=12, color='white')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate a Faster R-CNN network')
    parser.add_argument('prefix', help='saved model prefix', default="model/resnet_voc0712", type=str)
    parser.add_argument('epoch', help='epoch of pretrained model', default=10, type=int)
    parser.add_argument('gpu', help='GPU device to use', default=0, type=int)
    parser.add_argument('image', help='custom image', default="street_small.jpg", type=str)
    parser.add_argument('--vis', help='display result', action='store_true')
    args = parser.parse_args()
    return args


# parse args
args = parse_args()
ctx = mx.gpu(args.gpu)

# decode and resize image
im_orig = imdecode(args.image)
im, im_scale = resize(im_orig)

height, width = im.shape[:2]
im_info = mx.nd.array([height, width, im_scale])

# transform into tensor
im_tensor = transform(im)

# generate data batch
data_batch = generate_batch(im_tensor, im_info)

# assemble executor
symbol = get_resnet_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
predictor = get_net(symbol, args.prefix, args.epoch, ctx)

# forward
output = predictor.predict(data_batch)
rois = output['rois_output'][:, 1:]
scores = output['cls_prob_reshape_output'][0]
bbox_deltas = output['bbox_pred_reshape_output'][0]

# convert to per class detection results
cls = scores.argmax(axis=1, keepdims=True)
conf = scores.max(axis=1, keepdims=True)

box_deltas = pick_deltas(cls, bbox_deltas)
boxes = bbox_corner2center(rois)
boxes = bbox_decode(box_deltas, boxes)
pred_boxes = bbox_center2corner(boxes)

# post process box
data_dict = {dshape[0]: datum for dshape, datum in zip(data_batch.provide_data, data_batch.data)}
height, width, scale = data_dict['im_info'].asnumpy()[0]

pred_boxes = bbox_clip(pred_boxes, height, width)
pred_boxes = pred_boxes / scale

nms_in = mx.nd.concat(cls, conf, pred_boxes, dim=1)
nms_out = mx.nd.contrib.box_nms(nms_in, overlap_thresh=NMS_THRESH)
for [cls, conf, x1, y1, x2, y2] in nms_out.asnumpy():
    if cls > 0:
        print([cls, conf, x1, y1, x2, y2])

# if vis
if args.vis:
    vis_detection(im_orig.asnumpy(), nms_out.asnumpy(), CLASSES)
