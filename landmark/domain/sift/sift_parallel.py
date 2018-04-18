"""Class to compute SIFT on the test dataset and store the results"""

import os
import dask
import pandas as pd
import numpy as np
import csv
import cv2
from landmark.infrastructure.landmark_dataset import Landmark
from landmark.utils.configurable import Configurable

class AggregatedSiftImage(object):
    """Class to compute SIFT on an image and aggregate its results

    PARAMETERS
    ----------
    path: str
        path to the image to load
    """
    def __init__(self, path):
        self.path = path
        self.mean_sift = self._get_mean

    @property
    def _get_mean(self):
        """ 
        Method to compute the mean of the SIFT features

        RETURNS
        -------
        mean_sift: numpy array
            array of shape (,128) with the mean of the SIFT features
        """
        img = cv2.imread(self.path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        _, des = sift.detectAndCompute(gray, None)
        mean_sift = np.mean(des, axis=0)
        return mean_sift


class AggregatedSiftAlbum(object):
    """Class to compute the mean of the SIFT features for all the test dataset

    PARAMETERS
    ----------
    config_path: str
        path to the config file to find the data in the warehouse
    """

    def __init__(self, config_path):
        self.config_path = config_path
        self.test_data = Landmark(self.config_path).test
        self.indexes = list(self.test_data.index)
        self.warehouse = Configurable(self.config_path).config["data"]["warehouse"]
        self.directory = os.path.join(self.warehouse, "sift")

    def load(self, write=False):
        """Method to compute the mean of the SIFT fetaures in parallel and structure it
        
        PARAMETERS
        ----------
        write: boolean
            True to write the results (mean and exceptions)

        RETURNS
        -------
        structured_mean_sift_data: pandas dataframe
            each image id has 128 features aggregated from the results of SIFT
        exceptions: list of strings
            image id which have not been loaded
        """

        mean_sift_data, exceptions = _parallel_load(self.test_data)
        for exception in exceptions:
            self.indexes.remove(exception)
        structured_mean_sift_data = pd.DataFrame(mean_sift_data, index=self.indexes)
        structured_mean_sift_data.index.names = ["id"]

        if write:
            _ = self._write_results(structured_mean_sift_data, exceptions)

        return structured_mean_sift_data, exceptions
    

    def _write_results(self, data, exceptions):
        path_to_write_data = os.path.join(self.directory, "sift_mean.csv")
        path_to_write_exceptions = os.path.join(self.directory, "exceptions.csv")
        data.to_csv(path_to_write_data, index=True, sep=";")
        with open(path_to_write_exceptions, "w") as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(exceptions)
        return None

@dask.delayed
def _preprocess(ind, content):
    try:
        return AggregatedSiftImage(content["path"]).mean_sift
    except (FileNotFoundError, cv2.error) as e:
        print("File not found: %s" % content["path"])
        return ind

def _parallel_load(data):
    mean_results = [_preprocess(ind, content) for ind, content in data.iterrows()]
    mean_results = dask.compute(mean_results)[0]
    cleaned_results = [mean for mean in mean_results if isinstance(mean, str) is not True]
    exceptions = [mean for mean in mean_results if isinstance(mean, str)]
    return cleaned_results, exceptions
