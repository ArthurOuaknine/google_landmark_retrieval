"""Class to load and preprocess images"""

import dask
import numpy as np
import cv2

class LandmarkImage(object):
    """Class to load and preprocess an image

    PARAMETERS
    ----------
    path: string
        path to image
    
    RETURNS
    -------
    img: 3D numpy array
        Preprocessed and resized image
    """

    SIZE = (500, 500)

    def __init__(self, path):
        self.path = path
        self.img = self._load

    def preprocess(self):
    # TODO: image preprocessing on the self.img attribute
        pass

    @property
    def _load(self):
        img = cv2.imread(self.path)
        img = cv2.resize(img, dsize=self.SIZE, interpolation=cv2.INTER_CUBIC)
        return img


class LandmarkAlbum(object):
    """Class to load an preprocess a list of images in parallel

    PARAMETERS
    ----------
    paths: pandas dataframe
        dataframe with id as index and a path column

    RETURNS
    -------
    album: 4D numpy array
        Preprocessed and resized pool of images
    """

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
    except (FileNotFoundError, cv2.error) as e:
        print("File not found: %s" % path)

def _parallel_load(paths):
    album = [_preprocess(path) for path in paths["path"]]
    album = dask.compute(album)[0]
    cleaned_album = [img for img in album if img is not None]
    cleaned_album = np.expand_dims(cleaned_album, axis=0)[0]
    return cleaned_album
