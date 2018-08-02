import os
import pickle
from mxnet import gluon
from symnet.logger import logger


class Dataset(gluon.data.Dataset):
    classes = []

    def __init__(self, root):
        self._root = os.path.expanduser(root)
        self._cache_path = os.path.join(self._root, 'cache')

        if not os.path.isdir(self._root):
            raise OSError('Not initialized {}'.format(self._root))
        if not os.path.exists(self._cache_path):
            os.mkdir(self._cache_path)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def _get_cached(self, cache_item, fn, **kwargs):
        cache_path = os.path.join(self._cache_path, '{}.pkl'.format(cache_item))
        if os.path.exists(cache_path):
            logger.info('loading cache {}'.format(cache_path))
            with open(cache_path, 'rb') as fid:
                cached = pickle.load(fid)
            return cached
        else:
            logger.info('computing cache {}'.format(cache_path))
            cached = fn(**kwargs)
            logger.info('saving cache {}'.format(cache_path))
            with open(cache_path, 'wb') as fid:
                pickle.dump(cached, fid, pickle.HIGHEST_PROTOCOL)
            return cached
