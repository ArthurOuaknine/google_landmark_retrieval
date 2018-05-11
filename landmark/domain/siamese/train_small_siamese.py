import os
import pandas as pd
import numpy as np
import time
import json
from sklearn.model_selection import train_test_split

from landmark.infrastructure.similarity_dataset import SimilarityDataset
from landmark.domain.preprocessing.preprocessing_landmark_recognition import LandmarkRecognitionAlbum
from landmark.domain.siamese.small_siamese import SmallSiamese
from landmark.utils.batch import Batch
from landmark.utils.configurable import Configurable

if __name__=="__main__":
    directory = os.environ["LANDMARK_HOME"]
    path_to_config = os.path.join(directory, "config.ini")
    warehouse = Configurable(path_to_config).config["data"]["warehouse"]
    directory_to_init_file = "siamese/small_siamese/init_files"
    name_init_file = "init_small_siamese_testing.json"
    path_to_init_file = os.path.join(warehouse, directory_to_init_file, name_init_file)

    # Load an init file with all the mandatory parameters
    with open(path_to_init_file) as f:
        init_file = json.load(f)

    similarity_data_name = init_file["similarity_data"]
    data = SimilarityDataset(path_to_config, similarity_data_name).load
    X = data[["id1", "id2", "path1", "path2"]]
    y = data["similarity"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=42)
    nb_data_train = X_train.shape[0]

    batch_size = init_file["batch_size"]
    init_file["nb_data_train"] = nb_data_train
    nb_batch = int(np.ceil(nb_data_train/batch_size))
    nb_epoch = init_file["nb_epoch"]
    nb_iter = nb_epoch*nb_batch
    log_name = init_file["log_name"]


    train_batch = Batch(batch_size=batch_size, data=X_train, batch_start=0,
                        label=y_train, nb_epoch=nb_epoch)
    validation_batch = Batch(batch_size=batch_size, data=X_test, batch_start=0,
                             label=y_test, nb_epoch=1)
    siamese = SmallSiamese(path_to_config, init_file)
    
    for i in range(nb_iter):
        loss = siamese.train(train_batch, i, nb_iter)

        if i%20==0:
            _ = siamese.validation(validation_batch, i, nb_iter)

    _ = siamese.save(i)
