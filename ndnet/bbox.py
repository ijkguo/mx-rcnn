from mxnet import gluon

__all__ = ['BBoxCornerToCenter', 'BBoxCenterToCorner', 'BBoxClipper']


class BBoxCornerToCenter(gluon.HybridBlock):
    def __init__(self, axis=-1, split=False):
        super(BBoxCornerToCenter, self).__init__()
        self._split = split
        self._axis = axis

    def hybrid_forward(self, F, x):
        xmin, ymin, xmax, ymax = F.split(x, axis=self._axis, num_outputs=4)
        width = xmax - xmin
        height = ymax - ymin
        x = xmin + width / 2
        y = ymin + height / 2
        if not self._split:
            return F.concat(x, y, width, height, dim=self._axis)
        else:
            return x, y, width, height


class BBoxCenterToCorner(gluon.HybridBlock):
    def __init__(self, axis=-1, split=False):
        super(BBoxCenterToCorner, self).__init__()
        self._split = split
        self._axis = axis

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        x, y, w, h = F.split(x, axis=self._axis, num_outputs=4)
        hw = w / 2
        hh = h / 2
        xmin = x - hw
        ymin = y - hh
        xmax = x + hw
        ymax = y + hh
        if not self._split:
            return F.concat(xmin, ymin, xmax, ymax, dim=self._axis)
        else:
            return xmin, ymin, xmax, ymax


class BBoxClipper(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(BBoxClipper, self).__init__(**kwargs)

    def hybrid_forward(self, F, boxes, window, *args, **kwargs):
        boxes = F.maximum(boxes, 0.0)
        # window [B, 2] -> reverse hw -> tile [B, 4] -> [B, 1, 4], boxes [B, N, 4]
        m = F.tile(F.reverse(window, axis=1), reps=(2,)).reshape((0, -4, 1, -1))
        boxes = F.broadcast_minimum(boxes, m - 1)
        return boxes
