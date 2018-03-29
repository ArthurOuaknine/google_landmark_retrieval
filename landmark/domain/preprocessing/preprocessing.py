"""Class to load and preprocess images"""

import dask
import numpy as np
from PIL import Image

class LandmarkImage(object):
    """Class to load and preprocess an image"""
    
    def __init__(self, path):
        self.path = path
        self.image = self._load

    def preprocess(self):
        # TODO: image preprocessing
        pass

    @property
    def _load(self):
        img = Image.load(self.path)
        img = np.asarray(img)
        return im

class LandmarkAlbum(object):
    """Class to load an preprocess a list of images in parallel"""

    def __init__(self, paths):
        self.paths = paths
        self.album = self._parallel_load

    @property
    def load(self):
        self.album = self._parallel_load

    @dask.delayed
    def _get_path(self, path):
        return path

    @dask.delayed
    def _preprocess(self, path):
        # TODO: change return whith preprocessing
        # return LandmarkImage(path).preprocess
        return LandmarkImage(path).image

    @property
    def _parallel_load(self):
        all_paths = [self._get_path(path) for path in self.paths]
        album = [self._preprocess(path) for path in all_paths]
        album = dask.compute(album)[0]
        return np.array(album)
