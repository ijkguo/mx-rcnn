class VOC:
    def __init__(self, is_train):
        from gluoncv.data import VOCDetection
        from gluoncv.utils.metrics import VOC07MApMetric
        self._ds_cls = VOCDetection
        self._mt_cls = VOC07MApMetric
        self.default_imageset = '2007_trainval' if is_train else '2007_test'

    def set_args(self, args):
        args.rcnn_num_classes = len(self._ds_cls.CLASSES) + 1

    def get_dataset(self, imageset):
        splits = [(int(s.split('_')[0]), s.split('_')[1]) for s in imageset.split('+')]
        return self._ds_cls(splits=splits)

    def get_metric(self, dataset):
        return self._mt_cls(iou_thresh=0.5, class_names=dataset.classes)

    def get_names(self):
        return self._ds_cls.CLASSES


class COCO:
    def __init__(self, is_train):
        from gluoncv.data import COCODetection
        from gluoncv.utils.metrics import COCODetectionMetric
        self._is_train = is_train
        self._ds_cls = COCODetection
        self._mt_cls = COCODetectionMetric
        self.default_imageset = 'instances_train2017' if is_train else 'instances_val2017'

    def set_args(self, args):
        if self._is_train:
            args.lr = 0.00125
            args.wd = 1e-4
            args.lr_warmup = 8000
            args.epochs = 24
            args.lr_decay_epoch = '16,21'
            args.rcnn_batch_rois = 512
        else:
            args.rpn_post_nms_topk = 1000
            args.rcnn_batch_rois = 1000
            args.rcnn_nms_thresh = 0.5
        args.img_short_side = 800
        args.img_long_side = 1333
        args.rpn_anchor_scales = (2, 4, 8, 16, 32)
        args.rcnn_num_classes = len(self._ds_cls.CLASSES) + 1

    def get_dataset(self, imageset):
        splits = imageset.split('+')
        return self._ds_cls(splits=splits, skip_empty=self._is_train, use_crowd=not self._is_train)

    def get_metric(self, dataset):
        return self._mt_cls(dataset, save_prefix='coco', cleanup=True)

    def get_names(self):
        return self._ds_cls.CLASSES


class DatasetFactory:
    DATASETS = {
        'voc': VOC,
        'coco': COCO
    }
    def __init__(self, name):
        if name not in self.DATASETS:
            raise ValueError("dataset {} not supported".format(name))
        self._ds_cls = self.DATASETS[name]

    def get_train(self, args):
        ds = self._ds_cls(is_train=True)
        ds.set_args(args)
        imageset = args.imageset if args.imageset else ds.default_imageset
        dataset = ds.get_dataset(imageset)
        return dataset

    def get_test(self, args):
        ds = self._ds_cls(is_train=False)
        ds.set_args(args)
        imageset = args.imageset if args.imageset else ds.default_imageset
        dataset = ds.get_dataset(imageset)
        metric = ds.get_metric(dataset)
        return dataset, metric

    def get_demo(self, args):
        ds = self._ds_cls(is_train=False)
        ds.set_args(args)
        names = ds.get_names()
        return names
