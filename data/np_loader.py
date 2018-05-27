import mxnet as mx
import numpy as np

from data.np_image import get_image, tensor_vstack


def generate_batch(im_tensor, im_info):
    """return batch"""
    data = [im_tensor, im_info]
    data_shapes = [('data', im_tensor.shape), ('im_info', im_info.shape)]
    data_batch = mx.io.DataBatch(data=data, label=None, provide_data=data_shapes, provide_label=None)
    return data_batch


class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, batch_size, short, max_size, mean, std):
        super(TestLoader, self).__init__()

        # save parameters as properties
        self._roidb = roidb
        self._batch_size = batch_size
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std

        # infer properties from roidb
        self._size = len(self._roidb)
        self._index = np.arange(self._size)

        # decide data and label names (only for training)
        self._data_name = ['data', 'im_info']
        self._label_name = None

        # status variable
        self._cur = 0
        self._data = None
        self._label = None

        # get first batch to fill in provide_data and provide_label
        self.next()
        self.reset()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self._data_name, self._data)]

    @property
    def provide_label(self):
        return None

    def reset(self):
        self._cur = 0

    def iter_next(self):
        return self._cur + self._batch_size <= self._size

    def next(self):
        if self.iter_next():
            data_batch = mx.io.DataBatch(data=self.getdata(), label=self.getlabel(),
                                         pad=self.getpad(), index=self.getindex(),
                                         provide_data=self.provide_data, provide_label=self.provide_label)
            self._cur += self._batch_size
            return data_batch
        else:
            raise StopIteration

    def getdata(self):
        indices = self.getindex()
        im_tensorl, im_infol = [], []
        for index in indices:
            roi_rec = self._roidb[index]
            im_tensor, im_info, gt_boxes = get_image(roi_rec, self._short, self._max_size, self._mean, self._std)
            im_tensorl.append(im_tensor)
            im_infol.append(im_info)
        self._data = tensor_vstack(im_tensorl), tensor_vstack(im_infol)
        return self._data

    def getlabel(self):
        return None

    def getindex(self):
        cur_from = self._cur
        cur_to = min(cur_from + self._batch_size, self._size)
        return np.arange(cur_from, cur_to)

    def getpad(self):
        return max(self._cur + self.batch_size - self._size, 0)
