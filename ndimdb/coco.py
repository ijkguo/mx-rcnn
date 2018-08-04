import os
import mxnet as mx
import numpy as np
from .dataset import Dataset

from pycocotools.coco import COCO


class COCOSegmentation(Dataset):
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
               'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
               'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
               'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
               'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
               'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
               'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
               'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
               'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'coco'), splits=('val2017',)):
        super(COCOSegmentation, self).__init__(root)
        self.splits = splits
        self.coco_ind_to_class_ind, self.class_ind_to_coco_ind = \
            self._get_cached('map', self._load_map, split=splits[0])
        self.coco = []
        self.roidb = []
        for split in splits:
            _coco = self._get_cached('coco_' + split, self._load_coco, split=split)
            self.coco.append(_coco)
            self.roidb.extend(self._get_cached('roidb_' + split, self._load_roidb, split=split, coco=_coco))

    def __len__(self):
        return len(self.roidb)

    def __getitem__(self, idx):
        img_path = self.roidb[idx]['filename']
        img = mx.image.imread(img_path, 1)
        ids = self.roidb[idx]['classes']
        boxes = self.roidb[idx]['boxes']
        segms = self.roidb[idx]['segms']
        return img, np.hstack((boxes, ids)), segms

    def _load_coco(self, split):
        # example: annotations/instances_train2017.json
        anno_file = os.path.join(self._root, 'annotations', 'instances_{}.json').format(split)
        _coco = COCO(anno_file)
        return _coco

    def _load_map(self, split):
        _coco = self._get_cached('coco_' + split, self._load_coco, split=split)
        cats = [cat['name'] for cat in _coco.loadCats(_coco.getCatIds())]
        class_to_coco_ind = dict(zip(cats, _coco.getCatIds()))
        class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        coco_ind_to_class_ind = dict([(class_to_coco_ind[cls], class_to_ind[cls]) for cls in self.classes])
        class_ind_to_coco_ind = dict([(class_to_ind[cls], class_to_coco_ind[cls]) for cls in self.classes])
        return coco_ind_to_class_ind, class_ind_to_coco_ind

    def _load_roidb(self, split, coco):
        # example train2017/000000119993.jpg
        image_file_tmpl = os.path.join(self._root, split, '{}')

        image_ids = coco.getImgIds()
        roidb = []
        for image_id in image_ids:
            im_ann = coco.loadImgs(image_id)[0]
            filename = image_file_tmpl.format(im_ann['file_name'])
            width = im_ann['width']
            height = im_ann['height']

            annIds = coco.getAnnIds(imgIds=image_id, iscrowd=None)
            objs = coco.loadAnns(annIds)

            # sanitize bboxes
            valid_objs = []
            for obj in objs:
                if obj.get('ignore', 0) == 1:
                    continue
                x, y, w, h = obj['bbox']
                # xywh to xyxy
                x1, y1 = x, y
                x2, y2 = x1 + np.maximum(0, w), y1 + np.maximum(0, h)
                # clip to [0, w/h]
                x1 = np.minimum(width, np.maximum(0, x1))
                y1 = np.minimum(height, np.maximum(0, y1))
                x2 = np.minimum(width, np.maximum(0, x2))
                y2 = np.minimum(height, np.maximum(0, y2))
                # require non crowd objects, non-zero seg area and moe than 1x1 box size
                if obj['iscrowd'] == 0 and obj['area'] > 1 and x2 > x1 and y2 > y1 and (x2 - x1) * (y2 - y1) >= 4:
                    obj['clean_bbox'] = [x1, y1, x2, y2]
                    valid_objs.append(obj)

                    segs = obj['segmentation']
                    assert isinstance(segs, list)
                    valid_segs = [np.asarray(p).reshape(-1, 2).astype('float32') for p in segs if len(p) >= 6]
                    obj['segmentation'] = valid_segs
            objs = valid_objs
            num_objs = len(objs)

            # skip empty images
            if not num_objs:
                continue
            boxes = np.zeros((num_objs, 4), dtype=np.float32)
            classes = np.zeros((num_objs, 1), dtype=np.int32)
            segms = []
            for ix, obj in enumerate(objs):
                boxes[ix, :] = obj['clean_bbox']
                classes[ix, 0] = self.coco_ind_to_class_ind[obj['category_id']]
                segms.append(obj['segmentation'])

            roidb.append({'image_id': image_id,
                          'filename': filename,
                          'classes': classes,
                          'boxes': boxes,
                          'segms': segms})
        return roidb
