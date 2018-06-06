import argparse
import time

import mxnet as mx
from mxnet import autograd, gluon
from gluoncv import data as gdata

from data.np_loader import AnchorGenerator, AnchorSampler
from data.transform import RCNNGluonTrainTransform
from net.logger import logger
from net.net_resnet import FRCNNResNet
from net.symbol_resnet import get_feat_size


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


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
    # training
    parser.add_argument('--gpus', help='GPU device to train with', default='0', type=str)
    parser.add_argument('--pretrained', help='pretrained model params',
                        default='model/res50-converted-0000.params', type=str)
    parser.add_argument('--frequent', help='frequency of logger', default=20, type=int)
    parser.add_argument('--resume', help='resume model prefix', default='', type=str)
    parser.add_argument('--prefix', help='new model prefix', default='model/e2e', type=str)
    parser.add_argument('--begin_epoch', help='begin epoch of training, use with resume', default=0, type=int)
    parser.add_argument('--end_epoch', help='end epoch of training', default=20, type=int)
    parser.add_argument('--lr', help='base learning rate', default=0.001, type=float)
    parser.add_argument('--lr_step', help='learning rate steps (in epoch)', default='14', type=str)
    args = parser.parse_args()
    return args


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    new_batch = []
    for i, data in enumerate(batch):
        new_data = gluon.utils.split_and_load(data, ctx_list=ctx_list)
        new_batch.append(new_data)
    return new_batch


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')

    def update(self, labels, preds):
        # label: [rpn_label, rpn_weight]
        # preds: [rpn_cls_logits]
        rpn_label, rpn_weight = labels
        rpn_cls_logits = preds[0]

        # calculate num_inst (average on those fg anchors)
        num_inst = mx.nd.sum(rpn_weight)

        # cls_logits (b, c, h, w) red_label (b, 1, h, w)
        pred_label = mx.nd.argmax(rpn_cls_logits, axis=1, keepdims=True)
        # label (b, 1, h, w)
        num_acc = mx.nd.sum((pred_label == rpn_label) * rpn_weight)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += num_inst.asscalar()


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')

    def update(self, labels, preds):
        # label = [rpn_bbox_target, rpn_bbox_weight]
        # pred = [rpn_bbox_reg]
        rpn_bbox_target, rpn_bbox_weight = labels
        rpn_bbox_reg = preds[0]

        # calculate num_inst (average on those fg anchors)
        num_inst = mx.nd.sum(rpn_bbox_weight) / 4

        # calculate smooth_l1
        loss = mx.nd.sum(rpn_bbox_weight * mx.nd.smooth_l1(rpn_bbox_reg - rpn_bbox_target, scalar=3))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')

    def update(self, labels, preds):
        # label = [rcnn_label]
        # pred = [rcnn_cls]
        rcnn_label = labels[0]
        rcnn_cls = preds[0]

        # calculate num_acc
        pred_label = mx.nd.argmax(rcnn_cls, axis=1)
        num_acc = mx.nd.sum(pred_label == rcnn_label)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += rcnn_label.size


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')

    def update(self, labels, preds):
        # label = [rcnn_bbox_target, rcnn_bbox_weight]
        # pred = [rcnn_reg]
        rcnn_bbox_target, rcnn_bbox_weight = labels
        rcnn_bbox_reg = preds[0]

        # calculate num_inst
        num_inst = mx.nd.sum(rcnn_bbox_weight) / 4

        # calculate smooth_l1
        loss = mx.nd.sum(rcnn_bbox_weight * mx.nd.smooth_l1(rcnn_bbox_reg - rcnn_bbox_target, scalar=1))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()


def main():
    # print config
    args = parse_args()
    print('Called with argument: %s' % args)
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    batch_size = len(ctx)

    # load testing data
    train_dataset = gdata.VOCDetection(splits=[(2007, 'trainval')])
    ag = AnchorGenerator(feat_stride=RPN_FEAT_STRIDE, anchor_scales=RPN_ANCHOR_SCALES, anchor_ratios=RPN_ANCHOR_RATIOS)
    asp = AnchorSampler(allowed_border=RPN_ALLOWED_BORDER, batch_rois=RPN_BATCH_ROIS,
                        fg_fraction=RPN_FG_FRACTION, fg_overlap=RPN_FG_OVERLAP)
    train_transform = RCNNGluonTrainTransform(short=IMG_SHORT_SIDE, max_size=IMG_LONG_SIDE, mean=IMG_PIXEL_MEANS,
                                              std=IMG_PIXEL_STDS, ac=get_feat_size, ag=ag, asp=asp)
    train_loader = gdata.DetectionDataLoader(train_dataset.transform(train_transform),
                                             batch_size=batch_size, shuffle=True, last_batch="keep", num_workers=0)

    # load model
    net = FRCNNResNet(
        num_anchors=RPN_ANCHORS, anchor_scales=RPN_ANCHOR_SCALES, anchor_ratios=RPN_ANCHOR_RATIOS,
        rpn_feature_stride=RPN_FEAT_STRIDE, rpn_pre_topk=RPN_PRE_NMS_TOP_N, rpn_post_topk=RPN_POST_NMS_TOP_N,
        rpn_nms_thresh=RPN_NMS_THRESH, rpn_min_size=RPN_MIN_SIZE,
        num_classes=RCNN_CLASSES, rcnn_feature_stride=RCNN_FEAT_STRIDE,
        rcnn_pooled_size=RCNN_POOLED_SIZE)
    if args.resume.strip():
        net.load_params(args.resume.strip())
    else:
        net.load_params(args.pretrained, allow_missing=True, ignore_extra=True)
        net.collect_params(net.prefix + 'rpn.*|' + net.prefix + 'rcnn.*').initialize()
    net.collect_params().reset_ctx(ctx)

    # loss
    rpn_cls_loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1, sparse_label=True, from_logits=True, weight=1. / RPN_BATCH_ROIS)
    rpn_reg_loss = gluon.loss.HuberLoss(rho=1. / 9, weight=1. / RPN_BATCH_ROIS)
    rcnn_cls_loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1, sparse_label=True, from_logits=True, weight=1. / RCNN_BATCH_ROIS)
    rcnn_reg_loss = gluon.loss.HuberLoss(rho=1, weight=1. / RCNN_BATCH_ROIS)
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
    lr = args.lr
    lr_decay = 0.1
    lr_steps = [int(epoch) for epoch in args.lr_step.split(',')]
    logger.info('lr {} lr_decay {}'.format(lr, lr_steps))

    # optimizer
    select = ['rcnn', 'rpn', 'stage2_conv', 'stage3_conv', 'stage4_conv']
    select = '|'.join([net.prefix + s for s in select])
    trainer = gluon.Trainer(
        net.collect_params(select),
        'sgd',
        {'learning_rate': lr,
         'wd': 0.0005,
         'momentum': 0.9,
         'clip_gradient': 5})

    # training loop
    for epoch in range(args.begin_epoch, args.end_epoch):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        for metric in metrics:
            metric.reset()
        tic = time.time()
        btic = time.time()
        for i, batch in enumerate(train_loader):
            batch = split_and_load(batch, ctx_list=ctx)
            batch_size = len(batch[0])
            losses = []
            metric_losses = [[] for _ in metrics]
            with autograd.record():
                for data, im_info, gt_bboxes, rpn_label, rpn_weight, rpn_bbox_target, rpn_bbox_weight in zip(*batch):
                    rpn_cls, rpn_reg, rcnn_cls, rcnn_reg, rcnn_label, rcnn_bbox_target, rcnn_bbox_weight = net(data, im_info, gt_bboxes)
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
                    with autograd.pause():
                        rpn_acc_metric.update([rpn_label, rpn_weight], [rpn_cls])
                        rpn_bbox_metric.update([rpn_bbox_target, rpn_bbox_weight], [rpn_reg])
                        rcnn_acc_metric.update([rcnn_label], [rcnn_cls])
                        rcnn_bbox_metric.update([rcnn_bbox_target, rcnn_bbox_weight], [rcnn_reg])
                autograd.backward(losses)
                for metric, record in zip(metrics, metric_losses):
                    metric.update(0, record)
            trainer.step(batch_size)
            # (batch_end_callback) update metrics
            if args.frequent and not (i + 1) % args.frequent:
                msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics + metrics2])
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}'.format(
                    epoch, i + 1, batch_size / (time.time() - btic), msg))
            btic = time.time()

        # (epoch_end_callback) save model
        msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
        logger.info('[Epoch {}] Training cost: {:.3f}, {}'.format(
            epoch, (time.time() - tic), msg))
        net.save_params('{:s}_{:04d}.params'.format(args.prefix, epoch + 1))


if __name__ == '__main__':
    main()
