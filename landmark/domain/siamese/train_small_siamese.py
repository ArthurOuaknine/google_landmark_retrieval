import os
import pandas as pd
import numpy as np
from landmark.infrastructure.similarity_dataset import SimilarityDataset
from landmark.domain.preprocessing.preprocessing_landmark_recognition import LandmarkRecognitionAlbum
from landmark.domain.siamese.small_siamese import SmallSiamese
from landmark.utils.batch import Batch

if __name__=="__main__":
    directory = os.environ["LANDMARK_HOME"]
    path_to_config = os.path.join(directory, "config.ini")
    similarity_data_name = "clean_similarity_label10000_sample100.csv"
    data = SimilarityDataset(path_to_config, similarity_data_name).load
    siamese = SmallSiamese(path_to_config)

    nb_batch = 10000
    batch_size = int(np.ceil(data.shape[0]/nb_batch))
    nb_epoch = 1
    nb_iter = nb_epoch*nb_batch

    batcher = Batch(batch_size, data, nb_epoch=nb_epoch)

    for i in range(nb_iter):
        batch_data = batcher.batch_data
        batch_imgs, exceptions = LandmarkRecognitionAlbum(batch_data).load
        import ipdb; ipdb.set_trace()
