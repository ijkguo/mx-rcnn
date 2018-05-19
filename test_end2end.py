import argparse
import pprint

import mxnet as mx
import numpy as np

from net.config import *
from net.model import get_net
from rcnn.logger import logger
from rcnn.config import config, default
from rcnn.dataset import PascalVOC
from rcnn.io.image import get_image
from rcnn.symbol.symbol_resnet import get_resnet_test


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
              nms_thresh=0.7, conf_thresh=1e-3):
    """rois (nroi, 4), scores (nrois, nclasses), bbox_deltas (nrois, 4 * nclasses), im_info (3)"""
    from rcnn.processing.bbox_transform import bbox_pred, clip_boxes
    from rcnn.processing.nms import py_nms_wrapper

    rois = rois.asnumpy()
    scores = scores.asnumpy()
    bbox_deltas = bbox_deltas.asnumpy()

    im_info = im_info.asnumpy()
    height, width, scale = im_info

    # post processing
    pred_boxes = bbox_pred(rois, bbox_deltas)
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
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    # general
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    parser.add_argument('--image_set', help='image_set name', default=default.test_image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    # testing
    parser.add_argument('prefix', help='model to test with', default=default.rcnn_prefix, type=str)
    parser.add_argument('epoch', help='model to test with', default=default.rcnn_epoch, type=int)
    parser.add_argument('gpu', help='GPU device to test with', default=0, type=int)
    args = parser.parse_args()
    return args


def main():
    # print config
    args = parse_args()
    logger.info('Called with argument: %s' % args)
    pprint.pprint(config)
    ctx = mx.gpu(args.gpu)

    # load testing data
    imdb = eval(args.dataset)(args.image_set, args.root_path, args.dataset_path)
    roidb = imdb.gt_roidb()
    test_data = TestLoader(roidb, batch_size=1)

    # load model
    sym = get_resnet_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    predictor = get_net(sym, args.prefix, args.epoch, ctx)

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

        det = im_detect(rois, scores, bbox_deltas, im_info, nms_thresh=NMS_THRESH, conf_thresh=1e-3)

        for j in range(1, imdb.num_classes):
            indexes = np.where(det[:, 0] == j)[0]
            all_boxes[j][i] = np.concatenate((det[:, -4:], det[:, [1]]), axis=-1)[indexes, :]

    # evaluate model
    imdb.evaluate_detections(all_boxes)


if __name__ == '__main__':
    main()
