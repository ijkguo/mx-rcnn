import argparse
import mxnet as mx

from data.bbox import bbox_corner2center, bbox_center2corner, bbox_decode, bbox_clip, pick_deltas
from data.transform import load_test, generate_batch
from data.vis import vis_detection
from net.model import get_net
from net.config import *
from rcnn.symbol.symbol_resnet import get_resnet_test

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


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

# load single test
im_tensor, im_info, im_orig = load_test(args.image)

# generate data batch
data_batch = generate_batch(im_tensor, im_info)

# assemble executor
symbol = get_resnet_test(num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS)
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
