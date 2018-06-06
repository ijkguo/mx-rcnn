"""
General image database
An image database creates a list of relative image path called image_set_index and
transform index to absolute image path. As to training, it is necessary that ground
truth and proposals are mixed together for training.
roidb
basic format [image_index]
['image', 'height', 'width', 'flipped',
'boxes', 'gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'bbox_targets']
"""

from net.logger import logger
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle


class IMDB(object):
    def __init__(self, name, image_set, root_path, dataset_path):
        """
        basic information about an image database
        :param name: name of image database will be used for any output
        :param root_path: root path store cache and proposal data
        :param dataset_path: dataset path store images and image lists
        """
        self.name = name + '_' + image_set
        self.image_set = image_set
        self.root_path = root_path
        self.data_path = dataset_path

        # abstract attributes
        self.classes = []
        self.num_classes = 0
        self.image_set_index = []
        self.num_images = 0

        self.config = {}

    def image_path_from_index(self, index):
        raise NotImplementedError

    def gt_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, detections):
        raise NotImplementedError

    @property
    def cache_path(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        cache_path = os.path.join(self.root_path, 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def image_path_at(self, index):
        """
        access image at index in image database
        :param index: image index in image database
        :return: image path
        """
        return self.image_path_from_index(self.image_set_index[index])

    def append_flipped_images(self, roidb):
        """
        append flipped images to an roidb
        flip boxes coordinates, images will be actually flipped when loading into network
        :param roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        :return: roidb: [image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        logger.info('%s append flipped images to roidb' % self.name)
        assert self.num_images == len(roidb)
        for i in range(self.num_images):
            roi_rec = roidb[i]
            boxes = roi_rec['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = roi_rec['width'] - oldx2 - 1
            boxes[:, 2] = roi_rec['width'] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'image': roi_rec['image'],
                     'height': roi_rec['height'],
                     'width': roi_rec['width'],
                     'boxes': boxes,
                     'gt_classes': roidb[i]['gt_classes'],
                     'gt_overlaps': roidb[i]['gt_overlaps'],
                     'max_classes': roidb[i]['max_classes'],
                     'max_overlaps': roidb[i]['max_overlaps'],
                     'flipped': True}
            roidb.append(entry)

        self.image_set_index *= 2
        return roidb
