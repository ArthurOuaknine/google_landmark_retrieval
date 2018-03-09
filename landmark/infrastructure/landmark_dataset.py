"""Class to load landmark data"""
import os
import pandas as pd
from landmark.utils.configurable import Configurable

class Landmark(Configurable):
    """Class to load dataset from Google"""

    TRAIN_PATH = "images/train.csv"
    TEST_PATH = "images/test.csv"

    def __init__(self, path_to_config):
        super(Landmark, self).__init__(path_to_config)
        self.cls = self.__class__
        self.warehouse = self.config["data"]["warehouse"]

    @property
    def train(self):
        """Method to load train dataset.
        
        RETURNS
        -------
        train_data: pandas dataframe
        """
        train_file = os.path.join(self.warehouse, self.cls.TRAIN_PATH)
        train_data = pd.read_csv(train_file, sep=";", index_col="id")
        return train_data

    @property
    def test(self):
        """Method to load test dataset.

        RETURNS
        -------
        test_data: pandas dataframe
        """
        test_file = os.path.join(self.warehouse, self.cls.TEST_PATH)
        test_data = pd.read_csv(test_file, sep=";", index_col="id")
        return test_data
