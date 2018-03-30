import os
import numpy as np
import cv2
import pytest
from landmark.domain.preprocessing.preprocessing import LandmarkImage

def test_load():
    directory = os.environ["LANDMARK_HOME"]
    fake_img = "landmark/tests/unit/domain/preprocessing/fakedata.jpg"
    path_to_fake = os.path.join(directory, fake_img)

    size = (500, 500)
    im_loaded = cv2.imread(path_to_fake)
    expected = cv2.resize(im_loaded, dsize=size, interpolation=cv2.INTER_CUBIC)
    truth = LandmarkImage(path_to_fake).img
    assert expected.shape == truth.shape
    assert np.array_equal(truth, expected)
