import argparse
import ast
import pprint

import mxnet as mx

from gluon_dataset import DatasetFactory
from gluon_network import NetworkFactory
from nddata.transform import load_test
from nddata.vis import vis_detection
from symdata.anchor import AnchorGenerator


def demo_net(net, class_names, args):
    # print config
    print('called with args\n{}'.format(pprint.pformat(vars(args))))

    # setup context
    if args.gpu:
        ctx = mx.gpu(int(args.gpu))
    else:
        ctx = mx.cpu(0)

    # load model
    net.load_parameters(args.params)
    net.collect_params().reset_ctx(ctx)

    # load single test
    ag = AnchorGenerator(feat_stride=args.rpn_feat_stride,
                         anchor_scales=args.rpn_anchor_scales, anchor_ratios=args.rpn_anchor_ratios)
    im_tensor, anchors, im_info, im_orig = load_test(args.image, short=args.img_short_side, max_size=args.img_long_side,
                                                     mean=args.img_pixel_means, std=args.img_pixel_stds,
                                                     feat_stride=args.rpn_feat_stride, ag=ag)

    # forward
    im_tensor = im_tensor.as_in_context(ctx)
    anchors = anchors.as_in_context(anchors)
    im_info = im_info.as_in_context(ctx)

    ids, scores, bboxes = net(im_tensor, anchors, im_info)
    det = mx.nd.concat(ids, scores, bboxes, dim=-1)[0]

    # remove background class
    det[:, 0] -= 1
    # scale back images
    det[:, 2:6] /= im_info[:, 2]

    # print out
    for [cls, conf, x1, y1, x2, y2] in det.asnumpy():
        if cls >= 0 and conf > args.vis_thresh:
            print(class_names[int(cls)], conf, [x1, y1, x2, y2])

    # if vis
    if args.vis:
        vis_detection(im_orig.asnumpy(), det.asnumpy(), class_names, thresh=args.vis_thresh)


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
    parser.add_argument('--rcnn-batch-rois', type=int, default=300)
    parser.add_argument('--rcnn-bbox-stds', type=str, default='(0.1, 0.1, 0.2, 0.2)')
    parser.add_argument('--rcnn-nms-thresh', type=float, default=0.3)
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
    class_names = DatasetFactory(args.dataset).get_demo(args)
    net = NetworkFactory(args.network).get_test(args)
    demo_net(net, class_names, args)


if __name__ == '__main__':
    main()
