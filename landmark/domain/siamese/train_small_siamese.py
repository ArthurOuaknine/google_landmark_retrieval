import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

from landmark.infrastructure.similarity_dataset import SimilarityDataset
from landmark.domain.preprocessing.preprocessing_landmark_recognition import LandmarkRecognitionAlbum
from landmark.domain.siamese.small_siamese import SmallSiamese
from landmark.utils.batch import Batch

def structure_data(batch_imgs, index_batch):
    batch = [batch_im[index_batch] for batch_im in batch_imgs]
    batch = np.expand_dims(batch, axis=0)[0]
    return batch

def structure_labels(data):
    labels = pd.DataFrame(data, columns=["similarity"])
    labels["similarity_env"] = 1 - labels["similarity"]
    labels = labels.as_matrix()
    return labels

if __name__=="__main__":
    directory = os.environ["LANDMARK_HOME"]
    path_to_config = os.path.join(directory, "config.ini")
    similarity_data_name = "clean_similarity_label10000_sample100.csv"
    data = SimilarityDataset(path_to_config, similarity_data_name).load
    X = data[["id1", "id2", "path1", "path2"]]
    y = data["similarity"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=42)

    batch_size = 64
    nb_batch = int(np.ceil(X_train.shape[0]/batch_size))
    nb_epoch = 1
    nb_iter = nb_epoch*nb_batch
    log_name = "second_try"

    train_batch = Batch(batch_size=batch_size, data=X_train, batch_start=0,
                        label=y_train, nb_epoch=nb_epoch)
    validation_batch = Batch(batch_size=batch_size, data=X_test, batch_start=0,
                             label=y_test, nb_epoch=1)
    siamese = SmallSiamese(path_to_config, log_name)
    
    for i in range(nb_iter):
        loss = siamese.train(train_batch, i, nb_iter)

        if i%20==0:
            _ = siamese.validation(validation_batch, i, nb_iter)
