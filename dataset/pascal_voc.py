import cv2
import os
import numpy as np

from net.logger import logger
from .imdb import IMDB
from .pascal_voc_eval import voc_eval


class PascalVOC(IMDB):
    def __init__(self, image_set, root_path, devkit_path):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'data', will write 'cache'
        :param devkit_path: 'data/VOCdevkit', load data and write results
        """
        super(PascalVOC, self).__init__('voc_' + image_set, root_path)
        self._classes = ['__background__',  # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor']

        year, image_set = image_set.split('_')
        self._config = {'comp_id': 'comp4',
                        'use_diff': False,
                        'min_size': 2}
        self._image_index_file = os.path.join(devkit_path, 'VOC' + year, 'ImageSets', 'Main', image_set + '.txt')
        self._image_file_tmpl = os.path.join(devkit_path, 'VOC' + year, 'JPEGImages', '{}.jpg')
        self._image_anno_tmpl = os.path.join(devkit_path, 'VOC' + year, 'Annotations', '{}.xml')

        # results
        result_folder = os.path.join(devkit_path, 'results', 'VOC' + year, 'Main')
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        self._result_file_tmpl = os.path.join(result_folder, 'comp4_det_' + image_set + '_{}.txt')

        # get roidb
        self._roidb = self._get_cached('roidb', self._load_gt_roidb)
        logger.info('%s num_images %d' % (self.name, self.num_images))

    def _load_gt_roidb(self):
        image_index = self._load_image_index()
        gt_roidb = [self._load_annotation(index) for index in image_index]
        return gt_roidb

    def _load_image_index(self):
        with open(self._image_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def _load_annotation(self, index):
        roi_rec = dict()
        # store image
        roi_rec['index'] = index
        roi_rec['image'] = self._image_file_tmpl.format(index)
        size = cv2.imread(roi_rec['image']).shape
        roi_rec['height'] = size[0]
        roi_rec['width'] = size[1]
        roi_rec['flipped'] = False

        # store original annotation
        objs = self._parse_voc_anno(self._image_anno_tmpl.format(index))
        roi_rec['objs'] = objs

        if not self._config['use_diff']:
            non_diff_objs = [obj for obj in objs if obj['difficult'] == 0]
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs,), dtype=np.int32)
        class_to_index = dict(zip(self.classes, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = obj['bbox'][0] - 1
            y1 = obj['bbox'][1] - 1
            x2 = obj['bbox'][2] - 1
            y2 = obj['bbox'][3] - 1
            cls = class_to_index[obj['name'].lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls

        roi_rec['boxes'] = boxes
        roi_rec['gt_classes'] = gt_classes
        return roi_rec

    @staticmethod
    def _parse_voc_anno(filename):
        import xml.etree.ElementTree as ET
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_dict = dict()
            obj_dict['name'] = obj.find('name').text
            obj_dict['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_dict['bbox'] = [int(float(bbox.find('xmin').text)),
                                int(float(bbox.find('ymin').text)),
                                int(float(bbox.find('xmax').text)),
                                int(float(bbox.find('ymax').text))]
            objects.append(obj_dict)
        return objects

    def _evaluate_detections(self, detections, use_07_metric=True, **kargs):
        self._write_pascal_results(detections)
        self._do_python_eval(detections, use_07_metric)

    def _write_pascal_results(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            logger.info('Writing %s VOC results file' % cls)
            filename = self._result_file_tmpl.format(cls)
            with open(filename, 'wt') as f:
                for im_ind, roi_rec in enumerate(self.roidb):
                    index = roi_rec['index']
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, all_boxes, use_07_metric):
        aps = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # class_anno is a dict [image_index, [bbox, difficult, det]]
            class_anno = {}
            npos = 0
            for roi_rec in self.roidb:
                index = roi_rec['index']
                objects = [obj for obj in roi_rec['objs'] if obj['name'] == cls]
                bbox = np.array([x['bbox'] for x in objects])
                difficult = np.array([x['difficult'] for x in objects]).astype(np.bool)
                det = [False] * len(objects)  # stand for detected
                npos = npos + sum(~difficult)
                class_anno[index] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

            # bbox is 2d array of all detections, corresponding to each image_id
            image_ids = []
            bbox = []
            confidence = []
            for im_ind, dets in enumerate(all_boxes[cls_ind]):
                for k in range(dets.shape[0]):
                    image_ids.append(self.roidb[im_ind]['index'])
                    bbox.append([dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1])
                    confidence.append(dets[k, -1])
            bbox = np.array(bbox)
            confidence = np.array(confidence)

            rec, prec, ap = voc_eval(class_anno, npos, image_ids, bbox, confidence,
                                     ovthresh=0.5, use_07_metric=use_07_metric)
            aps.append(ap)

        for cls, ap in zip(self.classes, aps):
            logger.info('AP for {} = {:.4f}'.format(cls, ap))
        logger.info('Mean AP = {:.4f}'.format(np.mean(aps)))
