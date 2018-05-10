import os
import pandas as pd
import numpy as np
import time
from landmark.infrastructure.similarity_dataset import SimilarityDataset
from landmark.domain.preprocessing.preprocessing_landmark_recognition import LandmarkRecognitionAlbum
from landmark.domain.siamese.small_siamese import SmallSiamese
from landmark.utils.batch import Batch

def structure_data(batch_imgs, index_batch):
    batch = [batch_im[index_batch] for batch_im in batch_imgs]
    batch = np.expand_dims(batch, axis=0)[0]
    return batch

def structure_labels(data):
    labels = pd.DataFrame(data["similarity"])
    labels.columns = ["similarity"]
    labels["similarity_env"] = 1 - labels["similarity"]
    labels = labels.as_matrix()
    return labels

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
        time1 = time.time()
        batch_data = batcher.batch_data
        batch_imgs, exceptions = LandmarkRecognitionAlbum(batch_data).load
        batch_x1 = structure_data(batch_imgs, 0)
        batch_x2 = structure_data(batch_imgs, 1)
        labels = structure_labels(batch_data)
        loss = siamese.train(batch_x1, batch_x2, labels, i)
        time2 = time.time()
        print("***** Loss at step %s: %s *****" %(i, str(loss)))
        print("===> Running time: %s" % (str(time2-time1)))
        print("**********")
        # TODO: split train/validation, create method for test and add accuracy to tensorboard
