"""Class to compute similarities using sift features"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from landmark.infrastructure.sift.sift_mean_dataset import SiftMean
from landmark.infrastructure.submission_format import Submission

class Similarities(object):
    """Class to find neighbors of an observation usinf aggregated sift features

    PARAMETERS
    ----------
    path_to_config: str
        path to config file

    """

    def __init__(self, path_to_config):
        self.path_to_config = path_to_config
        self.data = SiftMean(self.path_to_config).get_test
        self.exceptions = SiftMean(self.path_to_config).get_exceptions
        self.indexes = np.array(self.data.index)

    def neighborhood(self, n_neighbors):
        """Method to find the n neighbors of each observation and structure the results

        PARAMETERS
        ----------
        n_neighbors: int
            number of neighbors to compute

        RETURNS
        -------
        results: pandas dataframe
            dataframe with id of images as index and neighbors as columns
        """

        nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                                algorithm='ball_tree',
                                n_jobs=-1).fit(self.data)
        _, indices = nbrs.kneighbors(self.data)
        neighbors = self.indexes[indices]
        # neighbors = [[list(neighbor)] for neighbor in neighbors]
        cleaned_neighbors = list()
        for i in range(neighbors.shape[0]):
            neighbor = set(neighbors[i])
            neighbor.discard(self.indexes[i])
            cleaned_neighbors.append([list(neighbor)])
        results = pd.DataFrame(cleaned_neighbors, index=self.indexes)
        submission = Submission(results, self.exceptions)
        submission.export(self.path_to_config, file_name="sift_mean_similarities.csv")
        return results
