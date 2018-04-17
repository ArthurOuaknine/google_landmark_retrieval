"""Class to compute SIFT on the test dataset and store the results"""

import os
import pandas as pd
import numpy as np
import csv
import cv2
from landmark.infrastructure.landmark_dataset import Landmark
from landmark.utils.configurable import Configurable

class Sift(object):

    def __init__(self, config_path):
        self.config_path = config_path
        self.test_data = Landmark(self.config_path).test
        self.warehouse = Configurable(self.config_path).config["data"]["warehouse"]
        self.directory = os.path.join(self.warehouse, "sift")

    def get_mean(self, write=False):
        mean_results = []
        indexes = list(self.test_data.index)
        exception_indexes = []
        for ind, content in self.test_data.iterrows():
            try:
                img = cv2.imread(content["path"])
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sift = cv2.xfeatures2d.SIFT_create()
                _, des = sift.detectAndCompute(gray, None)
                mean_sift = np.mean(des, axis=0)
                mean_results.append(mean_sift)
            except cv2.error as e:
                print("Image loading error: %s" % content["path"])
                exception_indexes.append(ind)
                indexes.remove(ind)
        results = pd.DataFrame(mean_results, index=indexes)
        if write==True:
            path_to_write_data = os.path.join(self.directory, "sift_mean.csv")
            path_to_write_exceptions
            results.to_csv(path_to_write_data, index=True, sep=";")
            with open(path_to_write_exceptions, "wb") as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                wr.writerow(exception_indexes)
        return results
