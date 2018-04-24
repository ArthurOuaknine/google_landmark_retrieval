"""Class to load sift features"""

import glob
import os
import csv
import pandas as pd
from landmark.utils.configurable import Configurable

class SiftMean(Configurable):

    DATA_PATH = "sift/sift_mean_data"
    EXCEPTIONS_PATH = "sift/sift_mean_exceptions"

    def __init__(self, path_to_config):
        super(SiftMean, self).__init__(path_to_config)
        self.cls = self.__class__
        self.warehouse = self.config["data"]["warehouse"]

    @property
    def get_test(self):
        """Method to load the sift mean features for the test dataset"""
        path_to_data_files = os.path.join(self.warehouse, self.cls.DATA_PATH, "*.csv")
        data_files = glob.glob(path_to_data_files)
        test_data = pd.read_csv(data_files[0], sep=";", index_col="id")

        for i in range(1, len(data_files)):
            current_data = pd.read_csv(data_files[i], sep=";", index_col="id")
            test_data = pd.concat([test_data, current_data], axis=0)

        return test_data

    @property
    def get_exceptions(self):
        """Method to load the exceptions indices raised during SIFT computation"""
        path_to_data_files = os.path.join(self.warehouse, self.cls.EXCEPTIONS_PATH, "*.csv")
        data_files = glob.glob(path_to_data_files)
        exceptions = [self._load_exceptions_to_list(path) for path in data_files]
        flat_exceptions = [exception for sub_exceptions in exceptions for exception in sub_exceptions]
        return flat_exceptions

    def _load_exceptions_to_list(self, path_file):
        """Method to read a csv file containing exceptions and return a list of the ids"""
        with open(path_file, "r") as f:
            reader = csv.reader(f)
            exceptions = list(reader)
        return exceptions[0]
