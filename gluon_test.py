import argparse
import ast
import pprint

import mxnet as mx
from mxnet import gluon
from tqdm import tqdm

from gluon_dataset import DatasetFactory
from gluon_network import NetworkFactory
from nddata.transform import RCNNDefaultValTransform, batchify_append, batchify_pad, split_append, split_pad
from symdata.anchor import AnchorGenerator
from symnet.logger import logger


def test_net(net, dataset, metric, args):
    # print config
    logger.info('called with args\n{}'.format(pprint.pformat(vars(args))))

    # setup multi-gpu
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    batch_size = args.rcnn_batch_size * len(ctx)
    if args.rcnn_batch_size == 1:
        batchify_fn, split_fn = batchify_append, split_append
    else:
        batchify_fn, split_fn = batchify_pad, split_pad

    # load testing data
    ag = AnchorGenerator(feat_stride=args.rpn_feat_stride,
                         anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios)
    val_transform = RCNNDefaultValTransform(short=args.img_short_side, max_size=args.img_long_side,
                                            mean=args.img_pixel_means, std=args.img_pixel_stds,
                                            feat_stride=args.rpn_feat_stride, ag=ag)
    val_loader = gluon.data.DataLoader(dataset.transform(val_transform),
                                       batch_size=batch_size, shuffle=False, batchify_fn=batchify_fn,
                                       last_batch="keep", num_workers=4)

    # load model
    net.load_parameters(args.params)
    net.collect_params().reset_ctx(ctx)
    net.hybridize(static_alloc=True)

    # start detection
    with tqdm(total=len(dataset)) as pbar:
        for ib, batch in enumerate(val_loader):
            batch = split_fn(batch, ctx)

            # lazy eval
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []

            for data, anchors, im_info, label in zip(*batch):
                # append all labels
                gt_bboxes.append(label.slice_axis(axis=-1, begin=0, end=4))
                gt_ids.append(label.slice_axis(axis=-1, begin=4, end=5))
                gt_difficults.append(label.slice_axis(axis=-1, begin=5, end=6) if label.shape[-1] > 5 else None)

                # forward
                ids, scores, bboxes = net(data, anchors, im_info)
                # remove background class
                ids -= 1
                # scale back images
                bboxes /= im_info[:, 2]

                # append all results
                det_bboxes.append(bboxes)
                det_ids.append(ids)
                det_scores.append(scores)

            for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diffcult in zip(
                    det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults):
                metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diffcult)
            pbar.update(batch_size)
    names, values = metric.get()

    # print
    for k, v in zip(names, values):
        print(k, v)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str, default='resnet50', help='base network')
    parser.add_argument('--params', type=str, default='', help='path to trained model')
    parser.add_argument('--dataset', type=str, default='voc', help='training dataset')
    parser.add_argument('--imageset', type=str, default='', help='imageset splits')
    parser.add_argument('--gpus', type=str, default='0', help='gpu devices eg. 0,1')
    # faster rcnn params
    parser.add_argument('--img-short-side', type=int, default=600)
    parser.add_argument('--img-long-side', type=int, default=1000)
    parser.add_argument('--img-pixel-means', type=str, default='(0.0, 0.0, 0.0)')
    parser.add_argument('--img-pixel-stds', type=str, default='(1.0, 1.0, 1.0)')
    parser.add_argument('--rpn-feat-stride', type=int, default=16)
    parser.add_argument('--rpn-anchor-scales', type=str, default='(8, 16, 32)')
    parser.add_argument('--rpn-anchor-ratios', type=str, default='(0.5, 1, 2)')
    parser.add_argument('--rpn-pre-nms-topk', type=int, default=6000)
    parser.add_argument('--rpn-post-nms-topk', type=int, default=300)
    parser.add_argument('--rpn-nms-thresh', type=float, default=0.7)
    parser.add_argument('--rpn-min-size', type=int, default=16)
    parser.add_argument('--rcnn-num-classes', type=int, default=21)
    parser.add_argument('--rcnn-feat-stride', type=int, default=16)
    parser.add_argument('--rcnn-pooled-size', type=str, default='(14, 14)')
    parser.add_argument('--rcnn-batch-size', type=int, default=1)
    parser.add_argument('--rcnn-batch-rois', type=int, default=300)
    parser.add_argument('--rcnn-bbox-stds', type=str, default='(0.1, 0.1, 0.2, 0.2)')
    parser.add_argument('--rcnn-nms-thresh', type=float, default=0.3)
    parser.add_argument('--rcnn-nms-topk', type=int, default=-1)
    args = parser.parse_args()
    args.img_pixel_means = ast.literal_eval(args.img_pixel_means)
    args.img_pixel_stds = ast.literal_eval(args.img_pixel_stds)
    args.rpn_anchor_scales = ast.literal_eval(args.rpn_anchor_scales)
    args.rpn_anchor_ratios = ast.literal_eval(args.rpn_anchor_ratios)
    args.rcnn_pooled_size = ast.literal_eval(args.rcnn_pooled_size)
    args.rcnn_bbox_stds = ast.literal_eval(args.rcnn_bbox_stds)
    args.rcnn_batch_rois = args.rpn_post_nms_topk
    if not args.params:
        args.params = 'model/{}_{}_0020.params'.format(args.network, args.dataset)
    return args


def main():
    args = parse_args()
    dataset, metric = DatasetFactory(args.dataset).get_test(args)
    net = NetworkFactory(args.network).get_test(args)
    test_net(net, dataset, metric, args)


if __name__ == '__main__':
    main()
