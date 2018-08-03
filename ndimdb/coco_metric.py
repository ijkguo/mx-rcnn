import json
import mxnet as mx
import numpy as np

from .coco import COCOSegmentation
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as cocomask


class COCOSegmentationMetric(mx.metric.EvalMetric):
    def __init__(self, dataset: COCOSegmentation, result_filename):
        super(COCOSegmentationMetric, self).__init__('COCOMaskAP')
        self._dataset = dataset
        self._result_filename = result_filename
        self._current_id = 0
        self._all_results = []

    def update(self, bboxes, labels, scores, masks, *args, **kwargs):
        def _asnumpy(a):
            if a is None:
                return None
            elif isinstance(a, mx.nd.NDArray):
                return a.asnumpy()
            else:
                return a

        for bbox, label, score, mask in zip(*[_asnumpy(x) for x in [bboxes, labels, scores, masks]]):
            valid_pred = np.where((label >= 0) & (score >= 0.001))[0]
            label = label.flat[valid_pred].astype('int32')
            score = score.flat[valid_pred].astype('float32')

            bbox = bbox[valid_pred].astype('float32')
            bbox[:, 2] -= bbox[:, 0]
            bbox[:, 3] -= bbox[:, 1]

            mask = mask[valid_pred].astype('uint8')
            rle = cocomask.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
            rle['counts'] = rle['counts'].decode('ascii')

            res = {'image_id': self._dataset.roidb[self._current_id]['image_id'],
                   'category_id': self._dataset.class_ind_to_coco_ind[label],
                   'bbox': list(map(lambda x: float(round(x, 2)), bbox)),
                   'score': float(round(score, 3)),
                   'segmentation': rle}
            self._all_results.append(res)
            self._current_id += 1

    def get(self):
        with open(self._result_filename, 'w') as f:
            json.dump(self._all_results, f)

        coco = self._dataset.coco[0]
        cocoDt = coco.loadRes(self._result_filename)
        cocoEval = COCOeval(coco, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        cocoEval = COCOeval(coco, cocoDt, 'segm')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
