import argparse
import pprint

import mxnet as mx
import gluoncv as gcv

from ndnet.net_all import get_net
from nddata.transform import load_test
from nddata.vis import vis_detection


def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate a Faster R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str, default='resnet50_v2a', help='base network')
    parser.add_argument('--pretrained', type=str, default='', help='path to trained model')
    parser.add_argument('--dataset', type=str, default='voc', help='training dataset')
    parser.add_argument('--image', type=str, default='', help='path to test image')
    parser.add_argument('--gpu', type=str, default='', help='gpu device eg. 0')
    parser.add_argument('--vis', action='store_true', help='display results')
    parser.add_argument('--vis-thresh', type=float, default=0.7, help='threshold display boxes')
    parser.add_argument('--batch-images', type=int, default=1, help='batch size per gpu')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    class_names = get_class_names(args.dataset, args)
    net = get_net('_'.join((args.network, args.dataset)), args)
    demo_net(net, class_names, args)


def get_class_names(dataset, args):
    if dataset == 'voc':
        return gcv.data.VOCDetection.CLASSES
    elif dataset == 'coco':
        return gcv.data.COCODetection.CLASSES
    else:
        raise NotImplementedError('Dataset {} not implemented'.format(dataset))


def demo_net(net, class_names, args):
    # print config
    print('called with args\n{}'.format(pprint.pformat(vars(args))))

    # setup context
    if args.gpu:
        ctx = mx.gpu(int(args.gpu))
    else:
        ctx = mx.cpu(0)

    # load model
    net.load_parameters(args.pretrained)
    net.collect_params().reset_ctx(ctx)

    # load single test
    im_tensor, anchors, im_info, im_orig = load_test(
        args.image, short=net.img_short, max_size=net.img_max_size, mean=net.img_means, std=net.img_stds,
        anchors=net.anchors, asf=net.anchor_shape_fn)

    # forward
    im_tensor = im_tensor.as_in_context(ctx)
    anchors = anchors.as_in_context(ctx)
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


if __name__ == '__main__':
    main()
