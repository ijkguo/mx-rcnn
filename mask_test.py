import argparse
import pprint

import mxnet as mx
import numpy as np
from mxnet import gluon
from tqdm import tqdm

from ndimdb.coco import COCOSegmentation
from ndimdb.coco_metric import COCOSegmentationMetric
from nddata.transform import MaskDefaultValTransform
from ndnet.net_all import get_net
from symdata.mask import mask_resize_fill
from symnet.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Mask R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str, default='resnet50_v2a', help='base network')
    parser.add_argument('--pretrained', type=str, default='', help='path to trained model')
    parser.add_argument('--dataset', type=str, default='mask', help='training dataset')
    parser.add_argument('--imageset', type=str, default='', help='imageset splits')
    parser.add_argument('--gpus', type=str, default='0', help='gpu devices eg. 0,1')
    parser.add_argument('--batch-images', type=int, default=1, help='batch size per gpu')
    parser.add_argument('--num-workers', type=int, default=4, help='number of data loading workers')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # setup multi-gpu
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    batch_size = args.batch_images * len(ctx)

    # load model
    net = get_net('_'.join((args.network, args.dataset)), False, args)
    net.load_parameters(args.pretrained)
    net.collect_params().reset_ctx(ctx)

    # load testing data
    dataset, metric = get_dataset(args.dataset, args)
    val_loader = get_dataloader(net, dataset, batch_size, args)

    test_net(net, val_loader, metric, len(dataset), ctx, args)


def get_dataset(dataset, args):
    if dataset == 'mask':
        imageset = args.imageset if args.imageset else 'val2017'
        splits = imageset.split('+')
        val_dataset = COCOSegmentation(splits=splits)
        val_metric = COCOSegmentationMetric(val_dataset, imageset + '_seg.json')
    else:
        raise NotImplementedError('Dataset {} not implemented'.format(dataset))
    return val_dataset, val_metric


def get_dataloader(net, dataset, batch_size, args, shuffle=False):
    # load testing data
    val_transform = MaskDefaultValTransform(
        short=net.img_short, max_size=net.img_max_size, mean=net.img_means, std=net.img_stds,
        anchors=net.anchors, asf=net.anchor_shape_fn)
    val_loader = gluon.data.DataLoader(dataset.transform(val_transform),
        batch_size=batch_size, shuffle=shuffle, batchify_fn=net.batchify_fn, last_batch="keep", num_workers=args.num_workers)
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
            all_infos, all_ids, all_scores, all_bboxes, all_masks = [], [], [], [], []
            for data, anchors, im_info, label in zip(*batch):
                ids, scores, bboxes, masks = net(data, anchors, im_info)
                all_infos.append(im_info)
                all_ids.append(ids)
                all_scores.append(scores)
                all_bboxes.append(bboxes)
                all_masks.append(masks)

            for im_info, ids, scores, bboxes, masks in zip(all_infos, all_ids, all_scores, all_bboxes, all_masks):
                for i in range(im_info.shape[0]):
                    im_height, im_width, im_scale = im_info[i].asnumpy()
                    b_cls = ids[i].asnumpy()
                    b_conf = scores[i].asnumpy()
                    b_boxes = bboxes[i].asnumpy()
                    b_masks = masks[i].asnumpy()

                    valid = np.where(((b_cls >= 0) & (b_conf >= 0.001)))[0]
                    b_cls = b_cls[valid] - 1
                    b_conf = b_conf[valid]
                    b_boxes = b_boxes[valid] / im_scale
                    b_masks = b_masks[valid]

                    im_height, im_width = int(round(im_height / im_scale)), int(round(im_width / im_scale))
                    full_masks = []
                    for bbox, mask in zip(b_boxes, b_masks):
                        full_masks.append(mask_resize_fill(mask, bbox, (im_height, im_width)))
                    full_masks = np.array(full_masks)

                    metric.update(b_boxes, b_cls, b_conf, full_masks)
            pbar.update(batch_size)
    metric.get()


if __name__ == '__main__':
    main()
