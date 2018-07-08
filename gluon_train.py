import argparse
import ast
import pprint
import time

import mxnet as mx
from mxnet import autograd, gluon

from gluon_dataset import DatasetFactory
from gluon_network import NetworkFactory
from nddata.anchor import RPNTargetGenerator
from nddata.transform import RCNNDefaultTrainTransform, batchify_append, batchify_pad, split_append, split_pad
from ndnet.metric import RPNAccMetric, RPNL1LossMetric, RCNNAccMetric, RCNNL1LossMetric
from symdata.anchor import AnchorGenerator
from symnet.logger import logger


def main():
    args = parse_args()
    dataset = DatasetFactory(args.dataset).get_train(args)
    net, feat_shape_fn = NetworkFactory(args.network).get_train(args)

    # setup multi-gpu
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    batch_size = args.rcnn_batch_size * len(ctx)
    if args.dataset == 'coco' and batch_size > 1:
        args.lr *= batch_size
        args.lr_warmup /= batch_size
    else:
        args.lr_warmup = -1

    # load params
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
    else:
        net.load_parameters(args.pretrained, allow_missing=True, ignore_extra=True)
        net.collect_params('.*rpn|.*dense').initialize()
    net.collect_params().reset_ctx(ctx)

    # load training data
    train_loader, split_fn = get_dataloader(feat_shape_fn, dataset, batch_size, args)

    train_net(net, train_loader, split_fn, ctx, args)


def get_dataloader(feat_shape_fn, dataset, batch_size, args):
    if args.rcnn_batch_size == 1:
        batchify_fn, split_fn = batchify_append, split_append
    else:
        batchify_fn, split_fn = batchify_pad, split_pad

    # load training data
    ag = AnchorGenerator(feat_stride=args.rpn_feat_stride,
                         anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios)
    rtg = RPNTargetGenerator(num_sample=args.rpn_batch_rois, pos_iou_thresh=args.rpn_fg_overlap,
                             neg_iou_thresh=args.rpn_fg_overlap, pos_ratio=args.rpn_fg_fraction,
                             stds=(1.0, 1.0, 1.0, 1.0))
    train_transform = RCNNDefaultTrainTransform(short=args.img_short_side, max_size=args.img_long_side,
                                                mean=args.img_pixel_means, std=args.img_pixel_stds,
                                                feat_stride=args.rpn_feat_stride, ag=ag,
                                                asf=feat_shape_fn, rtg=rtg)
    train_loader = gluon.data.DataLoader(dataset.transform(train_transform),
                                         batch_size=batch_size, shuffle=True, batchify_fn=batchify_fn,
                                         last_batch="rollover", num_workers=4)
    return train_loader, split_fn


def get_lr_at_iter(alpha):
    return 1. / 3. * (1 - alpha) + alpha


def train_net(net: gluon.Block, train_loader, split_fn, ctx, args):
    # print config
    logger.info('called with args\n{}'.format(pprint.pformat(vars(args))))

    # loss
    rpn_cls_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(weight=1. / args.rpn_batch_rois)
    rpn_reg_loss = gluon.loss.HuberLoss(rho=1. / 9, weight=1. / args.rpn_batch_rois)
    rcnn_cls_loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1, sparse_label=True, weight=1. / args.rcnn_batch_rois)
    rcnn_reg_loss = gluon.loss.HuberLoss(rho=1, weight=1. / args.rcnn_batch_rois)
    metrics = [mx.metric.Loss('RPN_CE'),
               mx.metric.Loss('RPN_SmoothL1'),
               mx.metric.Loss('RCNN_CE'),
               mx.metric.Loss('RCNN_SmoothL1')]
    rpn_acc_metric = RPNAccMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    rcnn_acc_metric = RCNNAccMetric()
    rcnn_bbox_metric = RCNNL1LossMetric()
    metrics2 = [rpn_acc_metric, rpn_bbox_metric, rcnn_acc_metric, rcnn_bbox_metric]

    # learning rate
    lr_decay = 0.1
    lr_steps = [int(epoch) for epoch in args.lr_decay_epoch.split(',')]
    lr_warmup = float(args.lr_warmup)  # avoid int division

    # optimizer
    logger.info('training params\n{}'.format(pprint.pformat(list(net.collect_params(args.net_train_patterns).keys()))))
    logger.info('lr {} lr_decay {}'.format(args.lr, lr_steps))
    net.collect_params().setattr('grad_req', 'null')
    net.collect_params(args.net_train_patterns).setattr('grad_req', 'write')
    trainer = gluon.Trainer(
        net.collect_params(args.net_train_patterns),
        'sgd',
        {'learning_rate': args.lr,
         'wd': args.wd,
         'momentum': 0.9,
         'clip_gradient': 5})

    # training loop
    for epoch in range(args.start_epoch, args.epochs):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        for metric in metrics + metrics2:
            metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize(static_alloc=True)
        base_lr = trainer.learning_rate
        for i, batch in enumerate(train_loader):
            if epoch == 0 and i <= lr_warmup:
                # adjust based on real percentage
                new_lr = base_lr * get_lr_at_iter(i / lr_warmup)
                if new_lr != trainer.learning_rate:
                    if i % args.log_interval == 0:
                        logger.info('[Epoch 0 Iteration {}] Set learning rate to {}'.format(i, new_lr))
                    trainer.set_learning_rate(new_lr)
            batch = split_fn(batch, ctx)
            batch_size = len(batch[0])
            losses = []
            metric_losses = [[] for _ in metrics]
            add_losses = [[] for _ in metrics2]
            with autograd.record():
                for data, anchors, im_info, gt_bboxes, rpn_label, rpn_weight, rpn_bbox_target, rpn_bbox_weight in zip(*batch):
                    rpn_cls, rpn_reg, rcnn_cls, rcnn_reg, rcnn_label, rcnn_bbox_target, rcnn_bbox_weight = net(data, anchors, im_info, gt_bboxes)
                    # rpn loss
                    rpn_loss1 = rpn_cls_loss(rpn_cls, rpn_label, rpn_weight) * rpn_label.size / rpn_label.shape[0]
                    rpn_loss2 = rpn_reg_loss(rpn_reg, rpn_bbox_target, rpn_bbox_weight) * rpn_bbox_target.size / rpn_bbox_target.shape[0]
                    # rcnn loss
                    rcnn_loss1 = rcnn_cls_loss(rcnn_cls, rcnn_label) * rcnn_label.size / rcnn_label.shape[0]
                    rcnn_loss2 = rcnn_reg_loss(rcnn_reg, rcnn_bbox_target, rcnn_bbox_weight) * rcnn_bbox_target.size / rcnn_bbox_weight.shape[0]
                    # loss for backprop
                    losses.append(rpn_loss1.sum() + rpn_loss2.sum() + rcnn_loss1.sum() + rcnn_loss2.sum())
                    # loss for metrics
                    metric_losses[0].append(rpn_loss1.sum())
                    metric_losses[1].append(rpn_loss2.sum())
                    metric_losses[2].append(rcnn_loss1.sum())
                    metric_losses[3].append(rcnn_loss2.sum())
                    add_losses[0].append(([rpn_label, rpn_weight], [rpn_cls]))
                    add_losses[1].append(([rpn_bbox_target, rpn_bbox_weight], [rpn_reg]))
                    add_losses[2].append(([rcnn_label], [rcnn_cls]))
                    add_losses[3].append(([rcnn_bbox_target, rcnn_bbox_weight], [rcnn_reg]))
                autograd.backward(losses)
                for metric, record in zip(metrics, metric_losses):
                    metric.update(0, record)
                for metric, records in zip(metrics2, add_losses):
                    for record in records:
                        metric.update(record[0], record[1])
            trainer.step(batch_size)
            # (batch_end_callback) update metrics
            if args.log_interval and not (i + 1) % args.log_interval:
                msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics + metrics2])
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}'.format(
                    epoch, i + 1, args.log_interval * batch_size / (time.time() - btic), msg))
                btic = time.time()

        # (epoch_end_callback) save model
        msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
        logger.info('[Epoch {}] Training cost: {:.3f}, {}'.format(
            epoch, (time.time() - tic), msg))
        net.save_parameters('{:s}_{:04d}.params'.format(args.save_prefix, epoch + 1))


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str, default='resnet50', help='base network')
    parser.add_argument('--pretrained', type=str, default='', help='path to pretrained model')
    parser.add_argument('--dataset', type=str, default='voc', help='training dataset')
    parser.add_argument('--imageset', type=str, default='', help='imageset splits')
    parser.add_argument('--gpus', type=str, default='0', help='gpu devices eg. 0,1')
    parser.add_argument('--epochs', type=int, default=20, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--lr-decay-epoch', type=str, default='14', help='epoch to decay lr')
    parser.add_argument('--lr-warmup', type=str, default='', help='warmup iterations')
    parser.add_argument('--resume', type=str, default='', help='path to last saved model')
    parser.add_argument('--start-epoch', type=int, default=0, help='start epoch for resuming')
    parser.add_argument('--log-interval', type=int, default=100, help='logging mini batch interval')
    parser.add_argument('--save-prefix', type=str, default='', help='saving params prefix')
    # faster rcnn params
    parser.add_argument('--img-short-side', type=int, default=600)
    parser.add_argument('--img-long-side', type=int, default=1000)
    parser.add_argument('--img-pixel-means', type=str, default='(0.0, 0.0, 0.0)')
    parser.add_argument('--img-pixel-stds', type=str, default='(1.0, 1.0, 1.0)')
    parser.add_argument('--net-train-patterns', type=str, default='')
    parser.add_argument('--rpn-feat-stride', type=int, default=16)
    parser.add_argument('--rpn-anchor-scales', type=str, default='(8, 16, 32)')
    parser.add_argument('--rpn-anchor-ratios', type=str, default='(0.5, 1, 2)')
    parser.add_argument('--rpn-pre-nms-topk', type=int, default=12000)
    parser.add_argument('--rpn-post-nms-topk', type=int, default=2000)
    parser.add_argument('--rpn-nms-thresh', type=float, default=0.7)
    parser.add_argument('--rpn-min-size', type=int, default=16)
    parser.add_argument('--rpn-batch-rois', type=int, default=256)
    parser.add_argument('--rpn-allowed-border', type=int, default=0)
    parser.add_argument('--rpn-fg-fraction', type=float, default=0.5)
    parser.add_argument('--rpn-fg-overlap', type=float, default=0.7)
    parser.add_argument('--rpn-bg-overlap', type=float, default=0.3)
    parser.add_argument('--rcnn-num-classes', type=int, default=21)
    parser.add_argument('--rcnn-feat-stride', type=int, default=16)
    parser.add_argument('--rcnn-pooled-size', type=str, default='(14, 14)')
    parser.add_argument('--rcnn-batch-size', type=int, default=1)
    parser.add_argument('--rcnn-batch-rois', type=int, default=128)
    parser.add_argument('--rcnn-fg-fraction', type=float, default=0.25)
    parser.add_argument('--rcnn-fg-overlap', type=float, default=0.5)
    parser.add_argument('--rcnn-bbox-stds', type=str, default='(0.1, 0.1, 0.2, 0.2)')
    args = parser.parse_args()
    args.img_pixel_means = ast.literal_eval(args.img_pixel_means)
    args.img_pixel_stds = ast.literal_eval(args.img_pixel_stds)
    args.rpn_anchor_scales = ast.literal_eval(args.rpn_anchor_scales)
    args.rpn_anchor_ratios = ast.literal_eval(args.rpn_anchor_ratios)
    args.rcnn_pooled_size = ast.literal_eval(args.rcnn_pooled_size)
    args.rcnn_bbox_stds = ast.literal_eval(args.rcnn_bbox_stds)
    if not args.save_prefix:
        args.save_prefix = 'model/{}_{}'.format(args.network, args.dataset)
    return args


if __name__ == '__main__':
    main()
