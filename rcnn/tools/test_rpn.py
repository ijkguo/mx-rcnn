import argparse
import os
import mxnet as mx

from ..config import config
from ..symbol import *
from ..dataset import *
from ..core.loader import TestLoader
from ..core.tester import Predictor, generate_proposals
from ..utils.load_model import load_param

# rpn generate proposal config
config.TEST.HAS_RPN = True
config.TEST.RPN_PRE_NMS_TOP_N = -1
config.TEST.RPN_POST_NMS_TOP_N = 2000


def test_rpn(args, ctx, prefix, epoch,
             vis=False, shuffle=False, thresh=0):
    # load symbol
    sym = eval('get_' + args.network + '_rpn_test')()

    # load dataset and prepare imdb for training
    imdb = eval(args.dataset)(args.image_set, args.root_path, args.dataset_path)
    roidb = imdb.gt_roidb()
    test_data = TestLoader(roidb, batch_size=1, shuffle=shuffle, has_rpn=True)

    # load model
    arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx)

    # check parameters
    param_names = [k for k in sym.list_arguments() + sym.list_auxiliary_states()
                   if k not in dict(test_data.provide_data) and 'label' not in k]
    missing_names = [k for k in param_names if k not in arg_params and k not in aux_params]
    if len(missing_names):
        print 'detected missing params', missing_names

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data]
    label_names = [k[0] for k in test_data.provide_label]
    max_data_shape = [('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          test_data=test_data, arg_params=arg_params, aux_params=aux_params)

    # start testing
    imdb_boxes = generate_proposals(predictor, test_data, imdb, vis=vis, thresh=thresh)
    imdb.evaluate_recall(roidb, candidate_boxes=imdb_boxes)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Region Proposal Network')
    # general
    parser.add_argument('--network', help='network name',
                        default='vgg', type=str)
    parser.add_argument('--dataset', help='dataset name',
                        default='PascalVOC', type=str)
    parser.add_argument('--image_set', help='image_set name',
                        default='2007_test', type=str)
    parser.add_argument('--root_path', help='output data folder',
                        default='data', type=str)
    parser.add_argument('--dataset_path', help='dataset path',
                        default=os.path.join('data', 'VOCdevkit'), type=str)
    # testing
    parser.add_argument('--prefix', help='model to test with', type=str)
    parser.add_argument('--epoch', help='model to test with', type=int)
    # rpn
    parser.add_argument('--gpu', help='GPU device to test with', type=int)
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--thresh', help='rpn proposal threshold', default=0, type=float)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    ctx = mx.gpu(args.gpu)
    test_rpn(args, ctx, args.prefix, args.epoch,
             vis=args.vis, shuffle=args.shuffle, thresh=args.thresh)
