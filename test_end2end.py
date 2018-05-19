import argparse
import pprint

import mxnet as mx

from net.model import get_net
from rcnn.logger import logger
from rcnn.config import config, default
from rcnn.core.tester import pred_eval
from rcnn.core.loader import TestLoader
from rcnn.dataset import PascalVOC
from rcnn.symbol.symbol_resnet import get_resnet_test


def test_rcnn(args, ctx, prefix, epoch, thresh=1e-3):
    # print config
    logger.info(pprint.pformat(config))

    # load testing data
    imdb = eval(args.dataset)(args.image_set, args.root_path, args.dataset_path)
    roidb = imdb.gt_roidb()
    test_data = TestLoader(roidb, batch_size=1, shuffle=False, has_rpn=True)

    # load model
    symbol = get_resnet_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS)
    predictor = get_net(symbol, prefix, epoch, ctx)

    # start detection
    pred_eval(predictor, test_data, imdb, vis=False, thresh=thresh)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    # general
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    parser.add_argument('--image_set', help='image_set name', default=default.test_image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    # testing
    parser.add_argument('--prefix', help='model to test with', default=default.rcnn_prefix, type=str)
    parser.add_argument('--epoch', help='model to test with', default=default.rcnn_epoch, type=int)
    parser.add_argument('--gpu', help='GPU device to test with', default=0, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info('Called with argument: %s' % args)
    ctx = mx.gpu(args.gpu)
    test_rcnn(args, ctx, args.prefix, args.epoch)

if __name__ == '__main__':
    main()
