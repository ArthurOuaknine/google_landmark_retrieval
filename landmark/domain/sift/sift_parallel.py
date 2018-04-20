"""Class to compute SIFT on the test dataset and store the results"""

import os
import dask
import pandas as pd
import numpy as np
import csv
import cv2
from landmark.infrastructure.landmark_dataset import Landmark
from landmark.utils.configurable import Configurable
from landmark.utils.batch import Batch

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
        try:
            mean_sift = np.mean(des, axis=0)
            return mean_sift
        except IndexError:
            print("Error computing mean of SIFT features with the following image: %s" % self.path)


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
        self.nb_data = self.test_data.shape[0]
        self.warehouse = Configurable(self.config_path).config["data"]["warehouse"]
        self.directory = os.path.join(self.warehouse, "sift")

    def load(self, file_name=None):
        """Method to compute the mean of the SIFT fetaures in parallel and structure it
        Data has to be written in batch (memory/debugging issues) which is fixed to 1000.
        
        PARAMETERS
        ----------
        file_name: None by default. If the name is given (without .csv extension), the data are written
            True to write the results (mean and exceptions)

        RETURNS
        -------
        all_structured_mean_sift_data: pandas dataframe
            each image id has 128 features aggregated from the results of SIFT
        all_exceptions: list of strings
            image id which have not been loaded
        """
        nb_batch = 1000
        batch_size = int(np.ceil(self.nb_data/nb_batch))
        batch = Batch(batch_size, self.test_data)

        for i in range(nb_batch):
            print("***** Starting loading batch %s *****" %i)
            mean_sift_data, exceptions = _parallel_load(batch.batch_data)
            batch_indexes = list(batch.batch_data.index)
            for exception in exceptions:
                batch_indexes.remove(exception)
            structured_mean_sift_data = pd.DataFrame(mean_sift_data, index=batch_indexes)
            structured_mean_sift_data.index.names = ["id"]

            if isinstance(file_name, str):
                _ = self._write_results(structured_mean_sift_data, exceptions, file_name, i)

            if i == 0:
                all_structured_mean_sift_data = structured_mean_sift_data
                all_exceptions = exceptions
            else:
                all_structured_mean_sift_data = pd.concat([all_structured_mean_sift_data,
                                                           structured_mean_sift_data],
                                                          axis=1)
                all_exceptions = all_exceptions + exceptions
            print("***** End batch %s *****" %i)

            batch.next

        return all_structured_mean_sift_data, all_exceptions
    

    def _write_results(self, data, exceptions, file_name, batch):
        complete_name_data = file_name + "_" + str(batch) + ".csv"
        complete_name_exceptions = file_name + "_exceptions_" + str(batch) + ".csv"
        path_to_write_data = os.path.join(self.directory, "sift_mean_data", complete_name_data)
        path_to_write_exceptions = os.path.join(self.directory, "sift_mean_exceptions", complete_name_exceptions)
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
