"""Class to format results for submission"""

import os
import pandas as pd
from landmark.utils.configurable import Configurable


class Submission(object):
    """Format and create file for submission

    PARAMETERS
    ----------
    results: pandas dataframe
        dataframe with id in index and list of similar id in a column

    """

    def __init__(self, results):
        self.results = results
        self._structure

    @property
    def _structure(self):
        if self.results.shape[1] != 1:
            raise ValueError("Too much column in result data frame!")
        self.results.columns = ["images"]
        self.results["images"] = self.results["images"].apply(lambda x: " ".join(x))
        return None

    def export(self, config_file, file_name="unkwnown.csv"):
        warehouse = Configurable(config_file).config["data"]["warehouse"]
        submission = os.path.join(warehouse, "submission", file_name)
        self.results.to_csv(submission, sep=",")
        return None
    
