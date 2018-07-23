import argparse
import pprint

import mxnet as mx
import gluoncv as gcv
from mxnet import gluon
from tqdm import tqdm

from ndnet.net_all import get_net
from nddata.transform import RCNNDefaultValTransform
from symnet.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str, default='resnet50_v2a', help='base network')
    parser.add_argument('--pretrained', type=str, default='', help='path to trained model')
    parser.add_argument('--dataset', type=str, default='voc', help='training dataset')
    parser.add_argument('--imageset', type=str, default='', help='imageset splits')
    parser.add_argument('--gpus', type=str, default='0', help='gpu devices eg. 0,1')
    parser.add_argument('--save-json', action='store_true', help='save coco output json')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset, metric = get_dataset(args.dataset, args)
    net = get_net(args.network, args)

    # setup multi-gpu
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    batch_size = args.rcnn_batch_size * len(ctx)

    # load model
    net.load_parameters(args.params)
    net.collect_params().reset_ctx(ctx)

    # load testing data
    val_loader = get_dataloader(net, dataset, batch_size, args)
    test_net(net, val_loader, metric, len(dataset), ctx, args)


def get_dataset(dataset, args):
    if dataset == 'voc':
        imageset = args.imageset if args.imageset else '2007_test'
        splits = [(int(s.split('_')[0]), s.split('_')[1]) for s in imageset.split('+')]
        val_dataset = gcv.data.VOCDetection(splits=splits)
        val_metric = gcv.utils.metrics.VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset == 'coco':
        imageset = args.imageset if args.imageset else 'instances_val2017'
        splits = imageset.split('+')
        val_dataset = gcv.data.COCODetection(splits=splits, skip_empty=False, use_crowd=True)
        val_metric = gcv.utils.metrics.COCODetectionMetric(val_dataset, save_prefix='coco', cleanup=not args.save_json)
    else:
        raise NotImplementedError('Dataset {} not implemented'.format(dataset))
    return val_dataset, val_metric


def get_dataloader(net, dataset, batch_size, args):
    # load testing data
    val_transform = RCNNDefaultValTransform(
        short=net.img_short, max_size=net.img_max_size, mean=net.img_means, std=net.img_stds,
        anchors=net.anchors, asf=net.anchor_shape_fn)
    val_loader = gluon.data.DataLoader(dataset.transform(val_transform),
        batch_size=batch_size, shuffle=False, batchify_fn=net.batchify_fn, last_batch="keep", num_workers=4)
    return val_loader


def test_net(net, val_loader, metric, size, ctx, args):
    # print config
    logger.info('called with args\n{}'.format(pprint.pformat(vars(args))))

    # prepare network
    net.hybridize(static_alloc=True)

    # start detection
    split_fn = net.split_fn
    with tqdm(total=size) as pbar:
        for ib, batch in enumerate(val_loader):
            batch = split_fn(batch, ctx)
            batch_size = len(batch[0])

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


if __name__ == '__main__':
    main()
