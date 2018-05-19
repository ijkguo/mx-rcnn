import argparse
import mxnet as mx

from data.bbox import decode_detect
from data.transform import load_test, generate_batch
from data.vis import vis_detection
from net.model import get_net
from net.config import *
from rcnn.symbol.symbol_resnet import get_resnet_test

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
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
im_info = im_info[0]

# decode detection
det = decode_detect(rois, scores, bbox_deltas, im_info, NMS_THRESH)

# remove background class
det[:, 0] -= 1

# print out
CONF_THRESH = 0.7
for [cls, conf, x1, y1, x2, y2] in det.asnumpy():
    if cls >= 0 and conf > CONF_THRESH:
        print([cls, conf, x1, y1, x2, y2])

# if vis
if args.vis:
    vis_detection(im_orig.asnumpy(), det.asnumpy(), CLASSES, thresh=CONF_THRESH)
