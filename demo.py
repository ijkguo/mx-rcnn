import argparse
import mxnet as mx
import numpy as np

from rcnn.config import config
from rcnn.symbol.symbol_resnet import get_resnet_test
from rcnn.utils.load_model import load_param
from rcnn.core.tester import Predictor
from rcnn.processing.bbox_transform import bbox_pred, clip_boxes
from rcnn.processing.nms import py_nms_wrapper


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


def im_detect(predictor, data_batch):
    """use predictor"""
    # read input
    data_dict = {dshape[0]: datum for dshape, datum in zip(data_batch.provide_data, data_batch.data)}
    height, width, scale = data_dict['im_info'].asnumpy()[0]

    # save output
    output = predictor.predict(data_batch)
    rois = output['rois_output'].asnumpy()[:, 1:]
    scores = output['cls_prob_reshape_output'].asnumpy()[0]
    bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]

    # post processing
    pred_boxes = bbox_pred(rois, bbox_deltas)
    pred_boxes = clip_boxes(pred_boxes, (height, width))

    # we used scaled image to infer, so it is necessary to transform them back
    pred_boxes = pred_boxes / scale

    return scores, pred_boxes


def vis_all_detection(im_array, detections, class_names):
    """
    visualize all detections in one image
    :param im_array: [b=1 c h w] in rgb
    :param detections: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
    :param class_names: list of names in imdb
    :return:
    """
    import matplotlib.pyplot as plt
    import random
    plt.imshow(im_array)
    for j, name in enumerate(class_names):
        if name == '__background__':
            continue
        color = (random.random(), random.random(), random.random())  # generate a random color
        dets = detections[j]
        for det in dets:
            bbox = det[:4]
            score = det[-1]
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor=color, linewidth=3.5)
            plt.gca().add_patch(rect)
            plt.gca().text(bbox[0], bbox[1] - 2,
                           '{:s} {:.3f}'.format(name, score),
                           bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')
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
scores, pred_boxes = im_detect(predictor, data_batch)

# convert to per class detection results
nms = py_nms_wrapper(NMS_THRESH)
all_boxes = [[] for _ in CLASSES]
for cls in CLASSES:
    cls_ind = CLASSES.index(cls)
    cls_boxes = pred_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
    cls_scores = scores[:, cls_ind, np.newaxis]
    keep = np.where(cls_scores >= CONF_THRESH)[0]
    dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
    keep = nms(dets)
    all_boxes[cls_ind] = dets[keep, :]
boxes_this_image = [[]] + [all_boxes[j] for j in range(1, len(CLASSES))]

# print results
print('---class---')
print('[[x1, x2, y1, y2, confidence]]')
for ind, boxes in enumerate(boxes_this_image):
    if len(boxes) > 0:
        print('---%s---' % CLASSES[ind])
        print('%s' % boxes)

# if vis
if args.vis:
    vis_all_detection(im_orig.asnumpy(), boxes_this_image, CLASSES)
