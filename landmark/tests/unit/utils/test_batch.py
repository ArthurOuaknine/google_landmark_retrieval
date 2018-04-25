import os
import pytest
import pandas as pd
from landmark.utils.batch import Batch

PATH_TO_FAKE_DATA = "landmark/tests/unit/utils/fakedata.csv"
PATH_TO_FAKE_DATA2 = "landmark/tests/unit/utils/fakedata2.csv"

def test_next_1():
    directory = os.environ["LANDMARK_HOME"]
    path_to_data = os.path.join(directory, PATH_TO_FAKE_DATA)
    data = pd.read_csv(path_to_data, sep=';', index_col="id")
    batch_size = 2

    batch = Batch(batch_size, data)
    truth_batch = batch.batch_data
    expected_batch = data[:2]
    assert truth_batch.equals(expected_batch)

def test_next_2():
    directory = os.environ["LANDMARK_HOME"]
    path_to_data = os.path.join(directory, PATH_TO_FAKE_DATA)
    data = pd.read_csv(path_to_data, sep=';', index_col="id")
    batch_size = 2

    batch = Batch(batch_size, data, batch_start=0, label=None, nb_epoch=1)
    batch.next
    truth_batch = batch.batch_data
    expected_batch = pd.DataFrame(data.loc["id3"]).T
    assert truth_batch.equals(expected_batch)

def test_next_3():
    directory = os.environ["LANDMARK_HOME"]
    path_to_data = os.path.join(directory, PATH_TO_FAKE_DATA)
    data = pd.read_csv(path_to_data, sep=';', index_col="id")
    path_to_fake_batch = os.path.join(directory, PATH_TO_FAKE_DATA2)
    expected_batch = pd.read_csv(path_to_fake_batch, sep=";", index_col="id")
    batch_size = 2
    batch_start = 0

    batch = Batch(batch_size, data, batch_start=batch_start, label=None, nb_epoch=2)
    batch.next
    truth_batch = batch.batch_data
    assert truth_batch.equals(expected_batch)

def test_next_4():
    directory = os.environ["LANDMARK_HOME"]
    path_to_data = os.path.join(directory, PATH_TO_FAKE_DATA)
    data = pd.read_csv(path_to_data, sep=';', index_col="id")
    path_to_fake_batch = os.path.join(directory, PATH_TO_FAKE_DATA2)
    expected_batch = pd.read_csv(path_to_fake_batch, sep=";", index_col="id")
    batch_size = 2
    batch_start = 1

    batch = Batch(batch_size, data, batch_start=batch_start, label=None, nb_epoch=2)
    truth_batch = batch.batch_data
    assert truth_batch.equals(expected_batch)
