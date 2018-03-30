"""Class to load and preprocess images"""

import dask
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True # helps for trancated images

class LandmarkImage(object):
    """Class to load and preprocess an image"""
    SIZE = (200, 200, 3)

    def __init__(self, path):
        self.path = path
        self.img = self._load

    def preprocess(self):
    # TODO: image preprocessing on the self.img attribute
        pass

    @property
    def _load(self):
        img = Image.open(self.path)
        # img.thumbnail(self.SIZE, Image.ANTIALIAS)
        img = np.asarray(img)
        import ipdb; ipdb.set_trace()
        img = np.reshape(img, self.SIZE)
        return img


class LandmarkAlbum(object):
    """Class to load an preprocess a list of images in parallel"""

    def __init__(self, paths):
        self.paths = paths

    @property
    def load(self):
        album = _parallel_load(self.paths)
        return album

@dask.delayed
def _get_path(path):
    return path

@dask.delayed
def _preprocess(path):
    # TODO: change return whith preprocessing
    # return LandmarkImage(path).preprocess
    try:
        return LandmarkImage(path).img
    except FileNotFoundError:
        print("%s doesn't exists and cannot be loaded. Returning None." % path)

def _parallel_load(paths):
    # all_paths = [_get_path(path) for path in paths]
    album = [_preprocess(path) for path in paths["path"]]
    album = dask.compute(album)[0]
    return np.array(album)
