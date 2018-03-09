"""Class to load landmark data"""
import os
from landmark.utils.configurable import Configurable

class Landmark(Configurable):
    """Class to load dataset from Google"""

    TRAIN_DATA = "images/train.csv"
    TEST_DATA = "images/test.csv"

    def __init__(self, path_to_config):
        super(Landmark, self).__init__(path_to_config)
        self.landmark_retrieval_home = os.environ["LANDMARK_HOME"]
        self.warehouse = self.config["data"]["warehouse"]

