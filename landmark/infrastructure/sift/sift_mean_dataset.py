"""Class to load sift features"""

import glob
import os
import pandas as pd
from landmark.utils.configurable import Configurable

class SiftMean(Configurable):

    DATA_PATH = "sift/sift_mean_data"

    def __init__(self, path_to_config):
        super(SiftMean, self).__init__(path_to_config)
        self.cls = self.__class__
        self.warehouse = self.config["data"]["warehouse"]

    @property
    def test(self):
        """Method to load the sift mean features for the test dataset"""
        path_to_data_files = os.path.join(self.warehouse, self.cls.DATA_PATH, "*.csv")
        data_files = glob.glob(path_to_data_files)
        test_data = pd.read_csv(data_files[0], sep=";", index_col="id")

        for i in range(1, len(data_files)):
            current_data = pd.read_csv(data_files[i], sep=";", index_col="id")
            test_data = pd.concat([test_data, current_data], axis=0)

        return test_data
