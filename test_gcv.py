import argparse
import mxnet as mx

from gluoncv import data as gdata
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from tqdm import tqdm

from data.bbox import decode_detect
from data.transform import RCNNDefaultValTransform, generate_batch
from net.config import *
from net.model import get_net
from rcnn.symbol.symbol_resnet import get_resnet_test


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    # testing
    parser.add_argument('prefix', help='model to test with', type=str)
    parser.add_argument('epoch', help='model to test with', type=int)
    parser.add_argument('gpu', help='GPU device to test with', type=int)
    args = parser.parse_args()
    return args


def main():
    # print config
    args = parse_args()
    print('Called with argument: %s' % args)
    ctx = mx.gpu(args.gpu)

    # load testing data
    val_dataset = gdata.VOCDetection(splits=[(2007, 'test')])
    val_loader = gdata.DetectionDataLoader(val_dataset.transform(RCNNDefaultValTransform()),
                                           batch_size=1, shuffle=False, last_batch="keep", num_workers=4)

    # load model
    sym = get_resnet_test(num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS)
    predictor = get_net(sym, args.prefix, args.epoch, ctx)

    # start detection
    metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    with tqdm(total=len(val_dataset)) as pbar:
        for ib, batch in enumerate(val_loader):
            im_tensor, im_info, label = batch
            data_batch = generate_batch(im_tensor, im_info)

            gt_ids = label.slice_axis(axis=-1, begin=4, end=5)
            gt_bboxes = label.slice_axis(axis=-1, begin=0, end=4)
            gt_difficults = label.slice_axis(axis=-1, begin=5, end=6) if label.shape[-1] > 5 else None

            # forward
            im_info = im_info[0]
            output = predictor.predict(data_batch)
            rois = output['rois_output'][:, 1:]
            scores = output['cls_prob_reshape_output'][0]
            bbox_deltas = output['bbox_pred_reshape_output'][0]

            # post processing
            det = decode_detect(rois, scores, bbox_deltas, im_info, NMS_THRESH)
            cls = det.slice_axis(axis=-1, begin=0, end=1)
            conf = det.slice_axis(axis=-1, begin=1, end=2)
            boxes = det.slice_axis(axis=-1, begin=2, end=6)
            cls -= 1

            metric.update(boxes.expand_dims(0), cls.expand_dims(0), conf.expand_dims(0), gt_bboxes, gt_ids,
                          gt_difficults)
            pbar.update(batch[0].shape[0])
    names, values = metric.get()

    # print
    for k, v in zip(names, values):
        print(k, v)


if __name__ == '__main__':
    main()
