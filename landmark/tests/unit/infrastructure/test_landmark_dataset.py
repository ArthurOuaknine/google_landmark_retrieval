"""Test for Landmark class"""
import os
import configparser
import pytest
import pandas as pd
from landmark.infrastructure.landmark_dataset import Landmark

class TestLandmark(Landmark):
    """Create fake class to test methods """

    TRAIN_PATH = "landmark/tests/unit/infrastructure/fakedata.csv"
    TEST_PATH = "landmark/tests/unit/infrastructure/fakedata.csv"

    def __init__(self, directory):
        """directory is a path to a config file"""
        self.warehouse = os.environ["LANDMARK_HOME"]
        fake_config = configparser.ConfigParser()
        fake_config.read(directory)
        fake_config.set("data", "warehouse", self.warehouse)
        self.config = fake_config
        self.cls = self.__class__


FAKE_PATH = "landmark/tests/unit/infrastructure"
FAKE_DATA = "fakedata.csv"
LANDMARK_HOME = os.environ["LANDMARK_HOME"]
DIRECTORY = os.path.join(LANDMARK_HOME, FAKE_PATH, FAKE_DATA)
CONFIG_FILE = os.path.join(LANDMARK_HOME, "config.ini")

def test_train():
    """Test train class from Landmark class """
    truth = TestLandmark(CONFIG_FILE).train
    values = [["id1", "path1"],
              ["id2", "path2"],
              ["id3", "path3"]]
    columns = ["id", "path"]
    expected_results = pd.DataFrame(values, columns=columns)
    expected_results.set_index("id", inplace=True)
    assert expected_results.equals(truth)

