"""Class to create and loadA a similrity dataset from the Google Landmark Recognition Challenge"""

import os
import pandas as pd
import glob
from landmark.utils.configurable import Configurable

class SimilarityDataset(Configurable):
    """Class to generate and load a similarity (labelled) dataset"""

    TRAIN_PATH = "recognition_challenge/train"
    IMAGES_PATH = "recognition_images/train/images"

    def __init__(self, path_to_config, file_name=None):
        super(SimilarityDataset, self).__init__(path_to_config)
        self.cls = self.__class__
        self.warehouse = self.config["data"]["warehouse"]
        self.raw = self.config["data"]["raw"]
        self.path_to_images = os.path.join(self.raw, self.cls.IMAGES_PATH)
        self.file_name = file_name

    def generate(self, nb_label_to_use=1000, nb_sample_by_label=100):
        training_dataset = self._load_train
        sorted_labels = list(training_dataset["landmark_id"].value_counts().index) # labels ordered by occurrence (descending)
        top_sorted_labels = sorted_labels[:nb_label_to_use]
        filtered_training_dataset = training_dataset[training_dataset["landmark_id"].isin(top_sorted_labels)]
        similarity_data = list()

        for label_index in range(len(top_sorted_labels)):
            similar_data = filtered_training_dataset[filtered_training_dataset["landmark_id"] == top_sorted_labels[label_index]]
            dissimilar_data = filtered_training_dataset[filtered_training_dataset["landmark_id"] != top_sorted_labels[label_index]]
            for i in range(int(nb_sample_by_label/2)):
                similar_pair = self._find_pair(similar_data, i, True)
                similarity_data.append(similar_pair)
                dissimilar_pair = self._find_pair(dissimilar_data, i, False)
                similarity_data.append(dissimilar_pair)

            print("*** Labels Done: %s/%s ***" % (str(label_index+1), len(top_sorted_labels)))

        columns = ["id1", "id2", "similarity"]
        similarity_data = pd.DataFrame(similarity_data, columns=columns)
        similarity_data["path1"] = similarity_data["id1"].apply(lambda x: os.path.join(self.path_to_images, x + ".jpg"))
        similarity_data["path2"] = similarity_data["id2"].apply(lambda x: os.path.join(self.path_to_images, x + ".jpg"))

        if self.file_name is not None:
            self._write(similarity_data, self.file_name)

        return similarity_data

    @property
    def load(self):
        path_to_write = os.path.join(self.warehouse, self.cls.TRAIN_PATH, self.file_name)
        train_dataset = pd.read_csv(path_to_write, sep=";")
        return train_dataset

    def check_data_availability(self, data, name_clean_table=None):
        """Create method to check if the data are available and to create a new table"""
        available_images = glob.glob(os.path.join(self.path_to_images, "*.jpg"))
        clean_table = data[data["path1"].isin(available_images) & data["path2"].isin(available_images)]
        if name_clean_table is not None:
            self._write(clean_table, name_clean_table)
        return clean_table
    
    @property
    def _load_train(self):
        path = os.path.join(self.warehouse, self.cls.TRAIN_PATH, "train.csv")
        training_dataset = pd.read_csv(path, index_col="id")
        training_dataset.drop(["url"], axis=1, inplace=True)
        return training_dataset

    def _find_pair(self, data, seed, similar):
        sample = data.sample(2, random_state=seed)
        pair = list(sample.index)
        pair.append(int(similar))
        return pair

    def _write(self, data, file_name):
        path = os.path.join(self.warehouse, self.cls.TRAIN_PATH, file_name)
        data.to_csv(path, sep=";", index=False)
        return None
