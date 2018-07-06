from mxnet import gluon
from gluoncv.nn.bbox import BBoxCornerToCenter

__all__ = ['BBoxCornerToCenter', 'BBoxClipper']


class BBoxClipper(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(BBoxClipper, self).__init__(**kwargs)

    def hybrid_forward(self, F, boxes, window, *args, **kwargs):
        boxes = F.maximum(boxes, 0.0)
        # window [B, 2] -> reverse hw -> tile [B, 4] -> [B, 1, 4], boxes [B, N, 4]
        m = F.tile(F.reverse(window, axis=1), reps=(2,)).reshape((0, -4, 1, -1))
        boxes = F.broadcast_minimum(boxes, m)
        return boxes
