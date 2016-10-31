import mxnet as mx

from rcnn.tools.test_rcnn import parse_args
from rcnn.tools.test_rcnn import test_rcnn

if __name__ == '__main__':
    args = parse_args()
    ctx = mx.gpu(args.gpu)
    test_rcnn(args, ctx, args.prefix, args.epoch,
              vis=args.vis, shuffle=args.shuffle, has_rpn=args.has_rpn, proposal=args.proposal)
