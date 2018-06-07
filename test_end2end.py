import argparse

import mxnet as mx
import numpy as np

from data.np_bbox import im_detect
from data.np_loader import TestLoader
from dataset.pascal_voc import PascalVOC
from net.logger import logger
from net.model import get_net
from net.symbol_resnet import get_resnet_test


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
RCNN_BBOX_STDS = (0.1, 0.1, 0.2, 0.2)
RCNN_CONF_THRESH = 1e-3
RCNN_NMS_THRESH = 0.3


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
    # testing
    parser.add_argument('prefix', help='model to test with', default='model/e2e', type=str)
    parser.add_argument('epoch', help='model to test with', default=10, type=int)
    parser.add_argument('gpu', help='GPU device to test with', default=0, type=int)
    args = parser.parse_args()
    return args


def main():
    # print config
    args = parse_args()
    logger.info('Called with argument: %s' % args)
    ctx = mx.gpu(args.gpu)

    # load testing data
    imdb = PascalVOC("2007_test", "data", "data/VOCdevkit")
    test_data = TestLoader(imdb.roidb, batch_size=1, short=IMG_SHORT_SIDE, max_size=IMG_LONG_SIDE,
                           mean=IMG_PIXEL_MEANS, std=IMG_PIXEL_STDS)

    # load model
    sym = get_resnet_test(
        num_anchors=RPN_ANCHORS, anchor_scales=RPN_ANCHOR_SCALES, anchor_ratios=RPN_ANCHOR_RATIOS,
        rpn_feature_stride=RPN_FEAT_STRIDE, rpn_pre_topk=RPN_PRE_NMS_TOP_N, rpn_post_topk=RPN_POST_NMS_TOP_N,
        rpn_nms_thresh=RPN_NMS_THRESH, rpn_min_size=RPN_MIN_SIZE,
        num_classes=RCNN_CLASSES, rcnn_feature_stride=RCNN_FEAT_STRIDE,
        rcnn_pooled_size=RCNN_POOLED_SIZE, rcnn_batch_size=RCNN_BATCH_SIZE)
    predictor = get_net(sym, args.prefix, args.epoch, ctx,
                        short=IMG_SHORT_SIDE, max_size=IMG_LONG_SIDE)

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(imdb.num_images)]
                 for _ in range(imdb.num_classes)]

    # start detection
    for i, data_batch in enumerate(test_data):
        logger.info('testing %d/%d' % (i, imdb.num_images))

        im_info = data_batch.data[1][0]

        # forward
        output = predictor.predict(data_batch)
        rois = output['rois_output'][:, 1:]
        scores = output['cls_prob_reshape_output'][0]
        bbox_deltas = output['bbox_pred_reshape_output'][0]

        det = im_detect(rois, scores, bbox_deltas, im_info,
                        bbox_stds=RCNN_BBOX_STDS, nms_thresh=RCNN_NMS_THRESH, conf_thresh=RCNN_CONF_THRESH)
        for j in range(1, imdb.num_classes):
            indexes = np.where(det[:, 0] == j)[0]
            all_boxes[j][i] = np.concatenate((det[:, -4:], det[:, [1]]), axis=-1)[indexes, :]

    # evaluate model
    imdb.evaluate_detections(all_boxes)


if __name__ == '__main__':
    main()
