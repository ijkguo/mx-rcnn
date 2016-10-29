import mxnet as mx

from rcnn.tools.test_rcnn import parse_args
from rcnn.tools.test_rcnn import test_rcnn

if __name__ == '__main__':
    args = parse_args()
    ctx = mx.gpu(args.gpu)
    test_rcnn(ctx, args.prefix, args.epoch, args, args.has_rpn, args.proposal)
