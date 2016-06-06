import argparse
import logging
import os

import mxnet as mx

from rcnn.config import config
from rcnn.data_iter import ROIIter
from rcnn.solver import Solver
from rcnn.symbol import get_vgg_rcnn
from utils.load_data import load_ss_roidb
from utils.load_model import load_checkpoint, load_param
from utils.save_model import save_checkpoint

config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = True
config.TRAIN.FINETUNE = True


def train_net(image_set, year, root_path, devkit_path, pretrained, epoch,
              prefix, ctx, begin_epoch, end_epoch, frequent):
    # set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # load symbol
    sym = get_vgg_rcnn()

    # load training data
    voc, roidb, means, stds = load_ss_roidb(image_set, year, root_path, devkit_path, flip=True)
    train_data = ROIIter(roidb, batch_size=config.TRAIN.BATCH_IMAGES, shuffle=True, mode='train')

    # load pretrained
    args, auxs = load_param(pretrained, epoch, convert=True, ctx=ctx)

    # initialize params
    arg_shape, _, _ = sym.infer_shape(data=(1, 3, 224, 224), rois=(1, 5))
    arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
    args['cls_score_weight'] = mx.random.normal(mean=0, stdvar=0.01, shape=arg_shape_dict['cls_score_weight'], ctx=ctx)
    args['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'], ctx=ctx)
    args['bbox_pred_weight'] = mx.random.normal(mean=0, stdvar=0.001, shape=arg_shape_dict['bbox_pred_weight'], ctx=ctx)
    args['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'], ctx=ctx)

    # train
    solver = Solver(prefix, sym, ctx, begin_epoch, end_epoch, args, auxs, momentum=0.9, wd=0.0005,
                    learning_rate=1e-3, lr_scheduler=mx.lr_scheduler.FactorScheduler(30000, 0.1))
    solver.fit(train_data, frequent=frequent)

    # edit params and save
    for epoch in range(begin_epoch + 1, end_epoch + 1):
        arg_params, aux_params = load_checkpoint(prefix, epoch)
        arg_params['bbox_pred_weight'] = (arg_params['bbox_pred_weight'].T * mx.nd.array(stds, ctx=ctx)).T
        arg_params['bbox_pred_bias'] = arg_params['bbox_pred_bias'] * mx.nd.array(stds, ctx=ctx) + \
                                       mx.nd.array(means, ctx=ctx)
        save_checkpoint(prefix, epoch, arg_params, aux_params)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Region Proposal Network')
    parser.add_argument('--image_set', dest='image_set', help='can be trainval or train',
                        default='trainval', type=str)
    parser.add_argument('--year', dest='year', help='can be 2007, 2010, 2012',
                        default='2007', type=str)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default=os.path.join(os.getcwd(), 'data'), type=str)
    parser.add_argument('--devkit_path', dest='devkit_path', help='VOCdevkit path',
                        default=os.path.join(os.getcwd(), 'data', 'VOCdevkit'), type=str)
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'vgg16'), type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=1, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'rcnn'), type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',
                        default=0, type=int)
    parser.add_argument('--begin_epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=8, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=20, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = mx.gpu(args.gpu_id)
    train_net(args.image_set, args.year, args.root_path, args.devkit_path, args.pretrained, args.epoch,
              args.prefix, ctx, args.begin_epoch, args.end_epoch, args.frequent)
