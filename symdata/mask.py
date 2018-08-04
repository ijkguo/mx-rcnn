import cv2
import numpy as np
import pycocotools.mask as cocomask


def polys_to_mask(polys, height, width):
    """
    convert polygons to mask
    :param polys: list of np.ndarray (N, 2) of float
    :param height: int
    :param width: int
    :return: np.ndarray (height, width) of uint8
    """
    polys = [p.flatten().tolist() for p in polys]
    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle)


def polys_flip(polys, width, flip_x=False):
    """
    flip polygon by width
    :param polys: list of np.ndarray of float (N, 2)
    :param width: int
    :param flip_x: boolean
    :return: list of flipped polygon
    """
    def _flip(poly, width, flip_x):
        if flip_x:
            x = width - poly[:, 0]
            poly[:, 0] = x
        return poly
    return [_flip(p, width, flip_x) for p in polys]


def mask_resize_fill(box, mask, shape):
    """
    resize mask to size of box, then fill into full image
    :param box: tuple of float (x1, y1, x2, y2)
    :param mask: np.ndarray (M, M)
    :param shape: tuple of int (height, width)
    :return: np.ndarray (height, width) of uint8
    """
    x1, y1, x2, y2 = box
    x1, y1 = int(x1 + 0.5), int(y1 + 0.5)
    x2, y2 = int(x2 - 0.5), int(y2 - 0.5)
    x2, y2 = max(x1, x2), max(y1, y2)
    w, h = (x2 - x1 + 1), (y2 - y1 + 1)
    mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
    ret = np.zeros(shape, dtype='uint8')
    ret[y1:y2 + 1, x1:x2 + 1] = mask
    return ret
