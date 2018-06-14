import argparse
import mxnet as mx

from gluoncv import data as gdata

from nddata.bbox import decode_detect
from nddata.transform import load_test
from nddata.vis import vis_detection
from ndnet.net_resnet import FRCNNResNet


IMG_SHORT_SIDE = 600
IMG_LONG_SIDE = 1000
IMG_PIXEL_MEANS = (0.0, 0.0, 0.0)
IMG_PIXEL_STDS = (1.0, 1.0, 1.0)

RPN_ANCHORS = 9
RPN_ANCHOR_SCALES = (8, 16, 32)
RPN_ANCHOR_RATIOS = (0.5, 1, 2)
RPN_FEAT_STRIDE = 16
RPN_PRE_NMS_TOP_N = 6000
RPN_POST_NMS_TOP_N = 300
RPN_NMS_THRESH = 0.7
RPN_MIN_SIZE = 16

RCNN_CLASSES = 21
RCNN_FEAT_STRIDE = 16
RCNN_POOLED_SIZE = (14, 14)
RCNN_BATCH_SIZE = 1
RCNN_BATCH_ROIS = RPN_POST_NMS_TOP_N
RCNN_BBOX_STDS = (0.1, 0.1, 0.2, 0.2)
RCNN_NMS_THRESH = 0.3

VIS_THRESH = 0.7


def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate a Faster R-CNN network')
    parser.add_argument('params', help='model file path', default="model/resnet_voc0712", type=str)
    parser.add_argument('image', help='custom image', default="street_small.jpg", type=str)
    parser.add_argument('gpu', help='GPU device to use', default=0, type=int)
    parser.add_argument('--vis', help='display result', action='store_true')
    args = parser.parse_args()
    return args


def main():
    # parse args
    args = parse_args()
    ctx = mx.gpu(args.gpu)

    # load model
    net = FRCNNResNet(
        num_anchors=RPN_ANCHORS, anchor_scales=RPN_ANCHOR_SCALES, anchor_ratios=RPN_ANCHOR_RATIOS,
        rpn_feature_stride=RPN_FEAT_STRIDE, rpn_pre_topk=RPN_PRE_NMS_TOP_N, rpn_post_topk=RPN_POST_NMS_TOP_N,
        rpn_nms_thresh=RPN_NMS_THRESH, rpn_min_size=RPN_MIN_SIZE,
        num_classes=RCNN_CLASSES, rcnn_feature_stride=RCNN_FEAT_STRIDE, rcnn_pooled_size=RCNN_POOLED_SIZE,
        rcnn_batch_size=RCNN_BATCH_SIZE, rcnn_batch_rois=RCNN_BATCH_ROIS)
    net.load_params(args.params)
    net.collect_params().reset_ctx(ctx)

    # load single test
    im_tensor, im_info, im_orig = \
        load_test(args.image, short=IMG_SHORT_SIDE, max_size=IMG_LONG_SIDE, mean=IMG_PIXEL_MEANS, std=IMG_PIXEL_STDS)

    # forward
    im_tensor = im_tensor.as_in_context(ctx)
    im_info = im_info.as_in_context(ctx)

    rois, scores, bbox_deltas = net(im_tensor, im_info)
    rois = rois[:, 1:]
    scores = mx.nd.softmax(scores)
    im_info = im_info[0]

    # decode detection
    det = decode_detect(rois, scores, bbox_deltas, im_info, bbox_stds=RCNN_BBOX_STDS, nms_thresh=RCNN_NMS_THRESH)

    # remove background class
    det[:, 0] -= 1

    # print out
    for [cls, conf, x1, y1, x2, y2] in det.asnumpy():
        if cls >= 0 and conf > VIS_THRESH:
            print([cls, conf, x1, y1, x2, y2])

    # if vis
    if args.vis:
        vis_detection(im_orig.asnumpy(), det.asnumpy(), gdata.VOCDetection.CLASSES, thresh=VIS_THRESH)


if __name__ == '__main__':
    main()
