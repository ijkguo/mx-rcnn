import argparse
import logging
import os

import mxnet as mx

from rcnn.config import config
from rcnn.tools.train_rpn import train_rpn
from rcnn.tools.test_rpn import test_rpn
from rcnn.tools.train_rcnn import train_rcnn
from rcnn.utils.combine_model import combine_model


def alternate_train(args, ctx, pretrained, epoch, rpn_epoch, rcnn_epoch):
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # basic config
    begin_epoch = 0
    config.TRAIN.BG_THRESH_LO = 0.0

    logging.info('########## TRAIN RPN WITH IMAGENET INIT')
    config.TRAIN.HAS_RPN = True
    config.TRAIN.BATCH_SIZE = 1
    train_rpn(args, ctx, pretrained, epoch, 'model/rpn1', begin_epoch, rpn_epoch)

    logging.info('########## GENERATE RPN DETECTION')
    config.TEST.HAS_RPN = True
    config.TEST.RPN_PRE_NMS_TOP_N = -1
    config.TEST.RPN_POST_NMS_TOP_N = 2000
    test_rpn(args, ctx[0], 'model/rpn1', rpn_epoch)

    logging.info('########## TRAIN RCNN WITH IMAGENET INIT AND RPN DETECTION')
    config.TRAIN.HAS_RPN = False
    config.TRAIN.BATCH_SIZE = 128
    train_rcnn(args, ctx, pretrained, epoch, 'model/rcnn1', begin_epoch, rcnn_epoch)

    logging.info('########## TRAIN RPN WITH RCNN INIT')
    config.TRAIN.HAS_RPN = True
    config.TRAIN.BATCH_SIZE = 1
    train_rpn(args, ctx, 'model/rcnn1', rcnn_epoch, 'model/rpn2', begin_epoch, rpn_epoch,
              finetune=True)

    logging.info('########## GENERATE RPN DETECTION')
    config.TEST.HAS_RPN = True
    config.TEST.RPN_PRE_NMS_TOP_N = -1
    config.TEST.RPN_POST_NMS_TOP_N = 2000
    test_rpn(args, ctx[0], 'model/rpn2', rpn_epoch)

    logger.info('########## COMBINE RPN2 WITH RCNN1')
    combine_model('model/rpn2', rpn_epoch, 'model/rcnn1', rcnn_epoch, 'model/rcnn2', 0)

    logger.info('########## TRAIN RCNN WITH RPN INIT AND DETECTION')
    config.TRAIN.HAS_RPN = False
    config.TRAIN.BATCH_SIZE = 128
    train_rcnn(args, ctx, 'model/rcnn2', 0, 'model/rcnn2', begin_epoch, rcnn_epoch,
               finetune=True)

    logger.info('########## COMBINE RPN2 WITH RCNN2')
    combine_model('model/rpn2', rpn_epoch, 'model/rcnn2', rcnn_epoch, 'model/final', 0)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN Network')
    # general
    parser.add_argument('--network', help='network name',
                        default='vgg', type=str)
    parser.add_argument('--dataset', help='dataset name',
                        default='PascalVOC', type=str)
    parser.add_argument('--image_set', help='image_set name',
                        default='2007_test', type=str)
    parser.add_argument('--root_path', help='output data folder',
                        default='data', type=str)
    parser.add_argument('--dataset_path', help='dataset path',
                        default=os.path.join('data', 'VOCdevkit'), type=str)
    # training
    parser.add_argument('--frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--kvstore', help='the kv-store type',
                        default='device', type=str)
    parser.add_argument('--work_load_list', help='work load for different devices',
                        default=None, type=list)
    parser.add_argument('--flip', help='flip images', action='store_true', default=True)
    parser.add_argument('--resume', help='continue training', action='store_true')
    # alternate
    parser.add_argument('--gpus', help='GPU device to train with',
                        default='0', type=str)
    parser.add_argument('--pretrained', help='pretrained model prefix',
                        default=os.path.join('model', 'vgg16'), type=str)
    parser.add_argument('--epoch', help='epoch of pretrained model',
                        default=1, type=int)
    parser.add_argument('--rpn_epoch', dest='rpn_epoch', help='end epoch of rpn training',
                        default=8, type=int)
    parser.add_argument('--rcnn_epoch', dest='rcnn_epoch', help='end epoch of rcnn training',
                        default=8, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = [mx.gpu(int(i)) for i in args.gpu_ids.split(',')]
    alternate_train(args, ctx, args.pretrained, args.epoch, args.rpn_epoch, args.rcnn_epoch)
