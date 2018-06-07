import argparse
import pprint

import mxnet as mx

from data.np_loader import AnchorGenerator, AnchorSampler, AnchorLoader
from dataset.pascal_voc import PascalVOC
from net.logger import logger
from net.module import MutableModule
from net.model import load_param, infer_data_shape, check_shape, initialize_frcnn, get_fixed_params
from net.metric import RPNAccMetric, RPNLogLossMetric, RPNL1LossMetric, RCNNAccMetric, RCNNLogLossMetric, RCNNL1LossMetric
from net.symbol_resnet import get_resnet_train


IMG_SHORT_SIDE = 600
IMG_LONG_SIDE = 1000
IMG_PIXEL_MEANS = (0.0, 0.0, 0.0)
IMG_PIXEL_STDS = (1.0, 1.0, 1.0)
NET_FIXED_PARAM = ['conv0', 'stage1', 'gamma', 'beta']

RPN_ANCHORS = 9
RPN_ANCHOR_SCALES = (8, 16, 32)
RPN_ANCHOR_RATIOS = (0.5, 1, 2)
RPN_FEAT_STRIDE = 16
RPN_PRE_NMS_TOP_N = 12000
RPN_POST_NMS_TOP_N = 2000
RPN_NMS_THRESH = 0.7
RPN_MIN_SIZE = 16
RPN_BATCH_ROIS = 256
RPN_ALLOWED_BORDER = 0
RPN_FG_FRACTION = 0.5
RPN_FG_OVERLAP = 0.7
RPN_BG_OVERLAP = 0.3

RCNN_CLASSES = 21
RCNN_FEAT_STRIDE = 16
RCNN_POOLED_SIZE = (14, 14)
RCNN_BATCH_SIZE = 1
RCNN_BATCH_ROIS = 128
RCNN_FG_FRACTION = 0.25
RCNN_FG_OVERLAP = 0.5
RCNN_BBOX_STDS = (0.1, 0.1, 0.2, 0.2)
RCNN_NMS_THRESH = 0.3


def train_net(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch,
              lr=0.001, lr_step='5'):
    # load symbol
    sym = get_resnet_train(num_anchors=RPN_ANCHORS, anchor_scales=RPN_ANCHOR_SCALES, anchor_ratios=RPN_ANCHOR_RATIOS,
                           rpn_feature_stride=RPN_FEAT_STRIDE, rpn_pre_topk=RPN_PRE_NMS_TOP_N, rpn_post_topk=RPN_POST_NMS_TOP_N,
                           rpn_nms_thresh=RPN_NMS_THRESH, rpn_min_size=RPN_MIN_SIZE, rpn_batch_rois=RPN_BATCH_ROIS,
                           num_classes=RCNN_CLASSES, rcnn_feature_stride=RCNN_FEAT_STRIDE, rcnn_pooled_size=RCNN_POOLED_SIZE,
                           rcnn_batch_size=RCNN_BATCH_SIZE, rcnn_batch_rois=RCNN_BATCH_ROIS, rcnn_fg_fraction=RCNN_FG_FRACTION,
                           rcnn_fg_overlap=RCNN_FG_OVERLAP, rcnn_bbox_stds=RCNN_BBOX_STDS)
    feat_sym = sym.get_internals()['rpn_cls_score_output']

    # setup multi-gpu
    batch_size = len(ctx)

    # load dataset and prepare imdb for training
    image_sets = ['2007_trainval']
    roidb = []
    for iset in image_sets:
        imdb = PascalVOC(iset, "data", "data/VOCdevkit")
        imdb.append_flipped_images()
        roidb.extend(imdb.roidb)

    # load training data
    ag = AnchorGenerator(feat_stride=RPN_FEAT_STRIDE, anchor_scales=RPN_ANCHOR_SCALES, anchor_ratios=RPN_ANCHOR_RATIOS)
    asp = AnchorSampler(allowed_border=RPN_ALLOWED_BORDER, batch_rois=RPN_BATCH_ROIS,
                        fg_fraction=RPN_FG_FRACTION, fg_overlap=RPN_FG_OVERLAP)
    train_data = AnchorLoader(roidb, batch_size, IMG_SHORT_SIDE, IMG_LONG_SIDE, IMG_PIXEL_MEANS, IMG_PIXEL_STDS,
                              feat_sym, ag, asp, shuffle=True)

    # produce shape max possible
    _, out_shape, _ = feat_sym.infer_shape(data=(1, 3, IMG_SHORT_SIDE, IMG_LONG_SIDE))
    FEAT_HEIGHT, FEAT_WIDTH = out_shape[0][-2:]
    data_names = ['data', 'im_info', 'gt_boxes']
    label_names = ['label', 'bbox_target', 'bbox_weight']
    data_shapes = [('data', (batch_size, 3, IMG_SHORT_SIDE, IMG_LONG_SIDE)),
                   ('im_info', (batch_size, 3)),
                   ('gt_boxes', (batch_size, 100, 5))]
    label_shapes = [('label', (batch_size, 1, RPN_ANCHORS * FEAT_HEIGHT, FEAT_WIDTH)),
                    ('bbox_target', (batch_size, 4 * RPN_ANCHORS, FEAT_HEIGHT, FEAT_WIDTH)),
                    ('bbox_weight', (batch_size, 4 * RPN_ANCHORS, FEAT_HEIGHT, FEAT_WIDTH))]

    # print shapes
    data_shape_dict, out_shape_dict = infer_data_shape(sym, data_shapes + label_shapes)
    logger.info('max input shape\n%s' % pprint.pformat(data_shape_dict))
    logger.info('max output shape\n%s' % pprint.pformat(out_shape_dict))

    # load and initialize params
    if args.resume:
        arg_params, aux_params = load_param(prefix, begin_epoch)
    else:
        arg_params, aux_params = load_param(pretrained, epoch)
        arg_params, aux_params = initialize_frcnn(sym, data_shapes, arg_params, aux_params)

    # check parameter shapes
    check_shape(sym, data_shapes + label_shapes, arg_params, aux_params)

    # check fixed params
    fixed_param_names = get_fixed_params(sym, NET_FIXED_PARAM)
    logger.info('locking params\n%s' % pprint.pformat(fixed_param_names))

    # metric
    rpn_eval_metric = RPNAccMetric()
    rpn_cls_metric = RPNLogLossMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    eval_metric = RCNNAccMetric()
    cls_metric = RCNNLogLossMetric()
    bbox_metric = RCNNL1LossMetric()
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)

    # callback
    batch_end_callback = mx.callback.Speedometer(batch_size, frequent=args.frequent, auto_reset=False)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)

    # learning schedule
    base_lr = lr
    lr_factor = 0.1
    lr_epoch = [int(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / batch_size) for epoch in lr_epoch_diff]
    logger.info('lr %f lr_epoch_diff %s lr_iters %s' % (lr, lr_epoch_diff, lr_iters))
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(lr_iters, lr_factor)
    # optimizer
    optimizer_params = {'momentum': 0.9,
                        'wd': 0.0005,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': (1.0 / batch_size),
                        'clip_gradient': 5}

    # train
    mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                        logger=logger, context=ctx, work_load_list=args.work_load_list,
                        max_data_shapes=data_shapes, max_label_shapes=label_shapes,
                        fixed_param_names=fixed_param_names)
    mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
            batch_end_callback=batch_end_callback, kvstore=args.kvstore,
            optimizer='sgd', optimizer_params=optimizer_params,
            arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN network')
    # training
    parser.add_argument('--frequent', help='frequency of logger', default=20, type=int)
    parser.add_argument('--kvstore', help='the kv-store type', default='device', type=str)
    parser.add_argument('--work_load_list', help='work load for different devices', default=None, type=list)
    parser.add_argument('--resume', help='continue training', action='store_true')
    # e2e
    parser.add_argument('--gpus', help='GPU device to train with', default='0', type=str)
    parser.add_argument('--pretrained', help='pretrained model prefix', default='model/resnet-50', type=str)
    parser.add_argument('--pretrained_epoch', help='pretrained model epoch', default=0, type=int)
    parser.add_argument('--prefix', help='new model prefix', default='model/e2e', type=str)
    parser.add_argument('--begin_epoch', help='begin epoch of training, use with resume', default=0, type=int)
    parser.add_argument('--end_epoch', help='end epoch of training', default=10, type=int)
    parser.add_argument('--lr', help='base learning rate', default=0.001, type=float)
    parser.add_argument('--lr_step', help='learning rate steps (in epoch)', default='7', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info('Called with argument: %s' % args)
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    train_net(args, ctx, args.pretrained, args.pretrained_epoch, args.prefix, args.begin_epoch, args.end_epoch,
              lr=args.lr, lr_step=args.lr_step)


if __name__ == '__main__':
    main()
