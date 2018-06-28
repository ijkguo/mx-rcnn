import argparse
import ast
import pprint

import mxnet as mx
from mxnet.module import Module

from symdata.bbox import im_detect
from symdata.loader import load_test, generate_batch
from symdata.vis import vis_detection
from symimdb.dataset import get_dataset_demo
from symnet.model import load_param, check_shape, get_max_shape_test
from symnet.network import get_network_test


def demo_net(sym, class_names, args):
    # print config
    print('called with args\n{}'.format(pprint.pformat(vars(args))))

    # setup context
    if args.gpu:
        ctx = mx.gpu(int(args.gpu))
    else:
        ctx = mx.cpu(0)

    # load single test
    im_tensor, im_info, im_orig = load_test(args.image, short=args.img_short_side, max_size=args.img_long_side,
                                            mean=args.img_pixel_means, std=args.img_pixel_stds)

    # generate data batch
    data_batch = generate_batch(im_tensor, im_info)

    # load params
    arg_params, aux_params = load_param(args.params, ctx=ctx)

    # produce shape max possible
    data_names, label_names, data_shapes, label_shapes = get_max_shape_test(
        args.img_short_side, args.img_long_side, 1)

    # check shapes
    check_shape(sym, data_shapes, arg_params, aux_params)

    # create and bind module
    mod = Module(sym, data_names, label_names, context=ctx)
    mod.bind(data_shapes, label_shapes, for_training=False)
    mod.init_params(arg_params=arg_params, aux_params=aux_params)

    # forward
    mod.forward(data_batch)
    rois, scores, bbox_deltas = mod.get_outputs()
    rois = rois[:, 1:]
    scores = scores[0]
    bbox_deltas = bbox_deltas[0]
    im_info = im_info[0]

    # decode detection
    det = im_detect(rois, scores, bbox_deltas, im_info,
                    bbox_stds=args.rcnn_bbox_stds, nms_thresh=args.rcnn_nms_thresh,
                    conf_thresh=args.rcnn_conf_thresh)

    # print out
    for [cls, conf, x1, y1, x2, y2] in det:
        if cls > 0 and conf > args.vis_thresh:
            print(class_names[int(cls)], conf, [x1, y1, x2, y2])

    # if vis
    if args.vis:
        vis_detection(im_orig, det, class_names, thresh=args.vis_thresh)


def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate a Faster R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str, default='resnet50', help='base network')
    parser.add_argument('--params', type=str, default='', help='path to trained model')
    parser.add_argument('--dataset', type=str, default='voc', help='training dataset')
    parser.add_argument('--image', type=str, default='', help='path to test image')
    parser.add_argument('--gpu', type=str, default='', help='gpu device eg. 0')
    parser.add_argument('--vis', action='store_true', help='display results')
    parser.add_argument('--vis-thresh', type=float, default=0.7, help='threshold display boxes')
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
    parser.add_argument('--rcnn-bbox-stds', type=str, default='(0.1, 0.1, 0.2, 0.2)')
    parser.add_argument('--rcnn-nms-thresh', type=float, default=0.3)
    parser.add_argument('--rcnn-conf-thresh', type=float, default=1e-3)
    args = parser.parse_args()
    args.img_pixel_means = ast.literal_eval(args.img_pixel_means)
    args.img_pixel_stds = ast.literal_eval(args.img_pixel_stds)
    args.rpn_anchor_scales = ast.literal_eval(args.rpn_anchor_scales)
    args.rpn_anchor_ratios = ast.literal_eval(args.rpn_anchor_ratios)
    args.rcnn_pooled_size = ast.literal_eval(args.rcnn_pooled_size)
    args.rcnn_bbox_stds = ast.literal_eval(args.rcnn_bbox_stds)
    if not args.params:
        args.params = '{}_{}-0010.params'.format(args.network, args.dataset)
    return args


def main():
    args = parse_args()
    class_names = get_dataset_demo(args.dataset, args)
    sym = get_network_test(args.network, args)
    demo_net(sym, class_names, args)


if __name__ == '__main__':
    main()
