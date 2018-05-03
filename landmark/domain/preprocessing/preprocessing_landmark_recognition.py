"""Class to load and preprocess images"""

import dask
import numpy as np
import cv2

class LandmarkRecognitionImages(object):
    """Class to load and preprocess an image

    PARAMETERS
    ----------
    paths: tuple of strings
        path for two similar images
    ind: int
        index of the tuple
    
    """

    SIZE = (500, 500)

    def __init__(self, paths):
        self.paths = paths
        self.imgs = self._load

    def preprocess(self):
    # TODO: image preprocessing on the self.img attribute
        pass

    @property
    def _load(self):
        img1 = cv2.imread(self.paths[0])
        img1 = cv2.resize(img1, dsize=self.SIZE, interpolation=cv2.INTER_CUBIC)
        img2 = cv2.imread(self.paths[1])
        img2 = cv2.resize(img2, dsize=self.SIZE, interpolation=cv2.INTER_CUBIC)
        return (img1, img2)
            

class LandmarkRecognitionAlbum(object):
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

    def __init__(self, paths_dataset):
        self.paths_dataset = paths_dataset

    @property
    def load(self):
        album, exceptions = _parallel_load(self.paths_dataset)
        return album, exceptions

@dask.delayed
def _preprocess(ind, content):
    # TODO: change return whith preprocessing
    # return LandmarkImage(path).preprocess
    paths = (content["path1"], content["path2"])
    try:
        recognition = LandmarkRecognitionImages(paths)
        return recognition.imgs
    except (FileNotFoundError, cv2.error) as e:
        print("File not found in the following tuple: %s" % str(paths))
        return ind

def _parallel_load(paths_dataset):
    album = [_preprocess(ind, content) for ind, content in paths_dataset.iterrows()]
    album = dask.compute(album)[0]
    cleaned_album = [imgs for imgs in album if isinstance(imgs, tuple)]
    exceptions = [imgs for imgs in album if isinstance(imgs, int)]
    return cleaned_album, exceptions
