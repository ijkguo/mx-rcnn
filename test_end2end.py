import argparse

import mxnet as mx
import numpy as np

from net.model import get_net
from net.symbol_resnet import get_resnet_test

from rcnn.logger import logger
from rcnn.dataset import PascalVOC
from rcnn.io.image import get_image


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


class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, batch_size=1):
        super(TestLoader, self).__init__()

        # save parameters as properties
        self.roidb = roidb
        self.batch_size = batch_size

        # infer properties from roidb
        self.size = len(self.roidb)
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        self.data_name = ['data', 'im_info']
        self.label_name = None

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.get_batch()
        self.reset()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data)]

    @property
    def provide_label(self):
        return None

    def reset(self):
        self.cur = 0

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)

        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        imgs, roidb = get_image(roidb)
        im_array = imgs[0]
        im_info = np.array([roidb[0]['im_info']], dtype=np.float32)
        data = {'data': im_array, 'im_info': im_info}

        self.data = [mx.nd.array(data[name]) for name in self.data_name]


def im_detect(rois, scores, bbox_deltas, im_info,
              bbox_stds, nms_thresh, conf_thresh):
    """rois (nroi, 4), scores (nrois, nclasses), bbox_deltas (nrois, 4 * nclasses), im_info (3)"""
    from rcnn.processing.bbox_transform import bbox_pred, clip_boxes
    from rcnn.processing.nms import py_nms_wrapper

    rois = rois.asnumpy()
    scores = scores.asnumpy()
    bbox_deltas = bbox_deltas.asnumpy()

    im_info = im_info.asnumpy()
    height, width, scale = im_info

    # post processing
    pred_boxes = bbox_pred(rois, bbox_deltas, bbox_stds)
    pred_boxes = clip_boxes(pred_boxes, (height, width))

    # we used scaled image & roi to train, so it is necessary to transform them back
    pred_boxes = pred_boxes / scale

    # convert to per class detection results
    nms = py_nms_wrapper(nms_thresh)

    det = []
    for j in range(1, scores.shape[-1]):
        indexes = np.where(scores[:, j] > conf_thresh)[0]
        cls_scores = scores[indexes, j, np.newaxis]
        cls_boxes = pred_boxes[indexes, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores))
        keep = nms(cls_dets)

        cls_id = np.ones_like(cls_scores) * j
        det.append(np.hstack((cls_id, cls_scores, cls_boxes))[keep, :])

    # assemble all classes
    det = np.concatenate(det, axis=0)
    return det


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
    roidb = imdb.gt_roidb()
    test_data = TestLoader(roidb, batch_size=1)

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
