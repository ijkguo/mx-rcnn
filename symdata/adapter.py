import mxnet as mx

class AnchorIter(mx.io.DataIter):
    def __init__(self, batch_size, loader):
        super(AnchorIter, self).__init__(batch_size)
        self._loader = loader
        self._iter = iter(self._loader)

        # decide data and label names
        self._data_name = ['data', 'im_info', 'gt_boxes']
        self._label_name = ['label', 'bbox_target', 'bbox_weight']
        self._data = None
        self._label = None

        # get first batch to fill in provide_data and provide_label
        self.next()
        self.reset()

    @property
    def size(self):
        return len(self._loader)

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self._data_name, self._data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self._label_name, self._label)]

    def reset(self):
        self._iter = iter(self._loader)

    def next(self):
        data, anchors, im_info, gt_boxes, label, label_weight, bbox_target, bbox_weight = next(self._iter)
        self._data = [data, im_info, gt_boxes]
        self._label = [label, bbox_target, bbox_weight]
        return mx.io.DataBatch(data=self._data, label=self._label, pad=0, index=None,
                               provide_data=self.provide_data, provide_label=self.provide_label)
