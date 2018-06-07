"""
General image database
An image database creates a list of relative image path called image_set_index and
transform index to absolute image path. As to training, it is necessary that ground
truth and proposals are mixed together for training.
Main functions of IMDB includes:
_load_roidb
append_flipped_images
evaluate_detection
roidb is a list of roi_rec
roi_rec is a dict of keys ["image", "height", "width", "boxes", "gt_classes", "flipped"]
"""

from net.logger import logger
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle


class IMDB(object):
    def __init__(self, name, root_path):
        """
        basic information about an image database
        :param root_path: root path store cache and proposal data
        """
        self._name = name
        self._root_path = root_path

        # abstract attributes
        self._classes = []
        self._roidb = []

        # create cache
        cache_folder = os.path.join(self._root_path, 'cache')
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)

    @property
    def name(self):
        return self._name

    @property
    def classes(self):
        return self._classes

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def roidb(self):
        return self._roidb

    @property
    def num_images(self):
        return len(self._roidb)

    def append_flipped_images(self):
        """Only flip boxes coordinates, images will be flipped when loading into network"""
        logger.info('%s append flipped images to roidb' % self._name)
        roidb_flipped = []
        for roi_rec in self._roidb:
            boxes = roi_rec['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = roi_rec['width'] - oldx2 - 1
            boxes[:, 2] = roi_rec['width'] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            roi_rec_flipped = {'image': roi_rec['image'],
                               'height': roi_rec['height'],
                               'width': roi_rec['width'],
                               'boxes': boxes,
                               'gt_classes': roi_rec['gt_classes']}
            roidb_flipped.append(roi_rec_flipped)
        self._roidb.extend(roidb_flipped)

    def _get_cached(self, cache_item, fn):
        cache_path = os.path.join(self._root_path, 'cache', '{}_{}.pkl'.format(self._name, cache_item))
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as fid:
                cached = pickle.load(fid)
            logger.info('loading cache {}'.format(cache_path))
            return cached
        else:
            cached = fn()
            with open(cache_path, 'wb') as fid:
                pickle.dump(cached, fid, pickle.HIGHEST_PROTOCOL)
            logger.info('saving cache {}'.format(cache_path))
            return cached

    def _load_gt_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, detections):
        raise NotImplementedError
