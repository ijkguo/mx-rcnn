import mxnet as mx
import numpy as np
import minibatch


class ROIIter(mx.io.DataIter):
    def __init__(self, roidb, batch_size=2, shuffle=False, mode='train'):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :return: ROIIter
        """
        super(ROIIter, self).__init__()

        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        if self.mode != 'train':
            assert self.batch_size == 1

        self.cur = 0
        self.size = len(roidb)
        self.index = np.arange(self.size)
        self.num_classes = self.roidb[0]['gt_overlaps'].shape[1]

        self.batch = None
        self.data = None
        self.label = None
        self.get_batch()
        self.data_name = self.data.keys()
        self.label_name = self.label.keys()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in self.data.items()]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in self.label.items()]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        return self.batch_size - self.size % self.batch_size

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[i] for i in range(cur_from, cur_to)]
        self.data, self.label = minibatch.get_minibatch(roidb, self.num_classes, self.mode)


class AnchorLoader(mx.io.DataIter):
    def __init__(self, feat_sym, roidb, batch_size=1, shuffle=False, mode='train',
                 feat_stride=16, anchor_scales=(8, 16, 32), allowed_border=0):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :return: AnchorLoader
        """
        super(AnchorLoader, self).__init__()

        self.feat_sym = feat_sym
        self.roidb = roidb
        self.batch_size = batch_size
        assert self.batch_size == 1
        self.shuffle = shuffle
        self.mode = mode
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.allowed_border = allowed_border

        self.cur = 0
        self.size = len(roidb)
        self.index = np.arange(self.size)
        self.num_classes = self.roidb[0]['gt_overlaps'].shape[1]

        self.batch = None
        self.data = None
        self.label = None
        self.get_batch()
        self.data_name = self.data.keys()
        self.label_name = self.label.keys()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in self.data.items()]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in self.label.items()]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        return self.batch_size - self.size % self.batch_size

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[i] for i in range(cur_from, cur_to)]
        self.data, self.label = minibatch.get_minibatch(roidb, self.num_classes, self.mode)
        data_shape = {k: v.shape for k, v in self.data.items()}
        del data_shape['im_info']
        _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
        feat_shape = [int(i) for i in feat_shape[0]]
        if self.mode == 'train':
            self.label = minibatch.assign_anchor(feat_shape, self.label['gt_boxes'], self.data['im_info'],
                                                 self.feat_stride, self.anchor_scales, self.allowed_border)
            del self.data['im_info']
