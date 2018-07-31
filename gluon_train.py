import argparse
import pprint
import time

import mxnet as mx
import gluoncv as gcv
from mxnet import autograd, gluon

from ndnet.net_all import get_net
from nddata.transform import RCNNDefaultTrainTransform
from ndnet.metric import RPNAccMetric, RPNL1LossMetric, RCNNAccMetric, RCNNL1LossMetric
from symnet.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str, default='resnet50_v2a', help='base network')
    parser.add_argument('--pretrained', type=str, default='', help='path to pretrained model')
    parser.add_argument('--dataset', type=str, default='voc', help='training dataset')
    parser.add_argument('--imageset', type=str, default='', help='imageset splits')
    parser.add_argument('--gpus', type=str, default='0', help='gpu devices eg. 0,1')
    parser.add_argument('--epochs', type=str, default='', help='training epochs')
    parser.add_argument('--lr', type=str, default='', help='base learning rate')
    parser.add_argument('--wd', type=str, default='', help='weight decay')
    parser.add_argument('--lr-decay-epoch', type=str, default='', help='epoch to decay lr')
    parser.add_argument('--lr-warmup', type=str, default='', help='warmup iterations')
    parser.add_argument('--resume', type=str, default='', help='path to last saved model')
    parser.add_argument('--start-epoch', type=int, default=0, help='start epoch for resuming')
    parser.add_argument('--log-interval', type=int, default=100, help='logging mini batch interval')
    parser.add_argument('--save-prefix', type=str, default='', help='saving params prefix')
    parser.add_argument('--batch-images', type=int, default=1, help='batch size per gpu')
    parser.add_argument('--num-workers', type=int, default=4, help='number of data loading workers')
    args = parser.parse_args()
    args.pretrained = args.pretrained if args.pretrained else 'model/{}_0000.params'.format(args.network)
    args.save_prefix = args.save_prefix if args.save_prefix else 'model/{}_{}'.format(args.network, args.dataset)
    if args.dataset == 'voc':
        args.epochs = int(args.epochs) if args.epochs else 20
        args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '14,20'
        args.lr = float(args.lr) if args.lr else 0.001
        args.lr_warmup = args.lr_warmup if args.lr_warmup else -1
        args.wd = float(args.wd) if args.wd else 5e-4
    else:
        args.epochs = int(args.epochs) if args.epochs else 24
        args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '16,21'
        args.lr = float(args.lr) if args.lr else 0.00125
        args.lr_warmup = args.lr_warmup if args.lr_warmup else 8000
        args.wd = float(args.wd) if args.wd else 1e-4
        batch_size = len(args.gpus.split(',')) * args.batch_images
        if batch_size == 1:
            args.lr_warmup = -1
        else:
            args.lr *= batch_size
            args.lr_warmup /= batch_size
    return args


def main():
    args = parse_args()

    # setup multi-gpu
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    batch_size = args.batch_images * len(ctx)

    # load model
    if args.resume.strip():
        net = get_net('_'.join((args.network, args.dataset)), False, args)
        net.load_parameters(args.resume.strip())
    else:
        net = get_net('_'.join((args.network, args.dataset)), True, args)
        net.collect_params('.*rpn|.*dense').initialize()
    net.collect_params().reset_ctx(ctx)

    # load training data
    dataset = get_dataset(args.dataset, args)
    train_loader = get_dataloader(net, dataset, batch_size, args)

    train_net(net, train_loader, ctx, args)


def get_dataset(dataset, args):
    if dataset == 'voc':
        imageset = args.imageset if args.imageset else '2007_trainval'
        splits = [(int(s.split('_')[0]), s.split('_')[1]) for s in imageset.split('+')]
        train_dataset = gcv.data.VOCDetection(splits=splits)
    elif dataset == 'coco':
        imageset = args.imageset if args.imageset else 'instances_train2017'
        splits = imageset.split('+')
        train_dataset = gcv.data.COCODetection(splits=splits, skip_empty=True, use_crowd=False)
    else:
        raise NotImplementedError('Dataset {} not implemented'.format(dataset))
    return train_dataset


def get_dataloader(net, dataset, batch_size, args):
    # load training data
    train_transform = RCNNDefaultTrainTransform(
        short=net.img_short, max_size=net.img_max_size, mean=net.img_means, std=net.img_stds,
        anchors=net.anchors, asf=net.anchor_shape_fn, rtg=net.anchor_target)
    train_loader = gluon.data.DataLoader(dataset.transform(train_transform),
        batch_size=batch_size, shuffle=True, batchify_fn=net.batchify_fn, last_batch="rollover", num_workers=args.num_workers)
    return train_loader


def get_lr_at_iter(alpha):
    return 1. / 3. * (1 - alpha) + alpha


def train_net(net, train_loader, ctx, args):
    # print config
    logger.info('called with args\n{}'.format(pprint.pformat(vars(args))))

    # loss
    rpn_cls_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    rpn_reg_loss = gluon.loss.HuberLoss(rho=1. / 9)
    rcnn_cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    rcnn_reg_loss = gluon.loss.HuberLoss(rho=1)
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
    logger.info('training params\n{}'.format(pprint.pformat(list(net.collect_params(net.train_patterns).keys()))))
    logger.info('lr {} lr_decay {}'.format(args.lr, lr_steps))
    net.collect_params().setattr('grad_req', 'null')
    net.collect_params(net.train_patterns).setattr('grad_req', 'write')
    trainer = gluon.Trainer(
        net.collect_params(net.train_patterns),
        'sgd',
        {'learning_rate': args.lr,
         'wd': args.wd,
         'momentum': 0.9,
         'clip_gradient': 5})

    # training loop
    split_fn = net.split_fn
    net.hybridize(static_alloc=True)
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
                for data, anchors, im_info, gt_bboxes, rpn_label, rpn_bbox_target, rpn_bbox_weight in zip(*batch):
                    rpn_cls, rpn_reg, rcnn_cls, rcnn_reg, rcnn_label, rcnn_bbox_target, rcnn_bbox_weight = net(data, anchors, im_info, gt_bboxes)
                    # rpn loss
                    rpn_cls = rpn_cls.squeeze(axis=-1)
                    num_rpn_pos = (rpn_label >= 0).sum()
                    rpn_loss1 = rpn_cls_loss(rpn_cls, rpn_label, rpn_label >= 0) * rpn_label.size / rpn_label.shape[0] / num_rpn_pos
                    rpn_loss2 = rpn_reg_loss(rpn_reg, rpn_bbox_target, rpn_bbox_weight) * rpn_bbox_target.size / rpn_bbox_target.shape[0] / num_rpn_pos
                    # rcnn loss
                    num_rcnn_pos = (rcnn_label >= 0).sum()
                    rcnn_loss1 = rcnn_cls_loss(rcnn_cls, rcnn_label, rcnn_label >= 0) * rcnn_label.size / rcnn_label.shape[0] / num_rcnn_pos
                    rcnn_loss2 = rcnn_reg_loss(rcnn_reg, rcnn_bbox_target, rcnn_bbox_weight) * rcnn_bbox_target.size / rcnn_bbox_weight.shape[0] / num_rcnn_pos
                    # loss for backprop
                    losses.append(rpn_loss1.sum() + rpn_loss2.sum() + rcnn_loss1.sum() + rcnn_loss2.sum())
                    # loss for metrics
                    metric_losses[0].append(rpn_loss1.sum())
                    metric_losses[1].append(rpn_loss2.sum())
                    metric_losses[2].append(rcnn_loss1.sum())
                    metric_losses[3].append(rcnn_loss2.sum())
                    add_losses[0].append(([rpn_label, rpn_label >= 0], [rpn_cls]))
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


if __name__ == '__main__':
    main()
