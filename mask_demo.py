import argparse
import pprint

import mxnet as mx

from ndimdb.coco import COCOSegmentation
from nddata.transform import load_test
from nddata.vis import vis_detection_mask
from ndnet.net_all import get_net
from symdata.mask import mask_resize_fill


def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate a Mask R-CNN network',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--network', type=str, default='resnet50_v2a', help='base network')
    parser.add_argument('--pretrained', type=str, default='', help='path to trained model')
    parser.add_argument('--dataset', type=str, default='mask', help='training dataset')
    parser.add_argument('--image', type=str, default='', help='path to test image')
    parser.add_argument('--gpu', type=str, default='', help='gpu device eg. 0')
    parser.add_argument('--vis', action='store_true', help='display results')
    parser.add_argument('--vis-thresh', type=float, default=0.7, help='threshold display boxes')
    parser.add_argument('--batch-images', type=int, default=1, help='batch size per gpu')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # setup context
    if args.gpu:
        ctx = mx.gpu(int(args.gpu))
    else:
        ctx = mx.cpu(0)

    # load model
    net = get_net('_'.join((args.network, args.dataset)), False, args)
    net.load_parameters(args.pretrained)
    net.collect_params().reset_ctx(ctx)

    # load data
    class_names = get_class_names(args.dataset, args)

    demo_net(net, class_names, ctx, args)


def get_class_names(dataset, args):
    if dataset == 'mask':
        return COCOSegmentation.classes
    else:
        raise NotImplementedError('Dataset {} not implemented'.format(dataset))


def demo_net(net, class_names, ctx, args):
    # print config
    print('called with args\n{}'.format(pprint.pformat(vars(args))))

    # load single test
    im_tensor, anchors, im_info, im_orig = load_test(
        args.image, short=net.img_short, max_size=net.img_max_size, mean=net.img_means, std=net.img_stds,
        anchors=net.anchors, asf=net.anchor_shape_fn)

    # forward
    im_tensor = im_tensor.as_in_context(ctx)
    anchors = anchors.as_in_context(ctx)
    im_info = im_info.as_in_context(ctx)
    im_shape = im_orig.shape[:2]

    ids, scores, bboxes, masks = net(im_tensor, anchors, im_info)

    # remove background class
    ids -= 1
    # scale back images
    bboxes /= im_info[:, 2]

    # convert to numpy
    ids = ids.asnumpy()[0]
    scores = scores.asnumpy()[0]
    bboxes = bboxes.asnumpy()[0]
    masks = masks.asnumpy()[0]

    # print out
    full_masks = []
    for i in range(len(ids)):
        cls = ids[i][0]
        conf = scores[i][0]
        x1, y1, x2, y2 = bboxes[i]
        mask = masks[i]
        full_masks.append(mask_resize_fill(mask, (x1, y1, x2, y2), im_shape))
        if cls >= 0 and conf > args.vis_thresh:
            print(class_names[int(cls)], conf, [x1, y1, x2, y2])

    # if vis
    if args.vis:
        import matplotlib.pyplot as plt
        vis_detection_mask(im_orig.asnumpy(), bboxes, scores, ids, full_masks, class_names, thresh=args.vis_thresh)
        plt.show()


if __name__ == '__main__':
    main()
