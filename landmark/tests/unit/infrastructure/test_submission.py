import os
import pandas as pd
import pytest
from landmark.infrastructure.submission_format import Submission

def test_structure():
    index = ['a', 'b', 'c']
    data = {"images":[['333', '444', '555'], ['666', '777'], ['999', '333', '111', '09']]}
    results = pd.DataFrame(data=data, index=index)
    results.index.name = "id"

    # results["images"] = results["images"].apply(lambda x: " ".join(x))
    truth = Submission(results).results

    home = os.environ["LANDMARK_HOME"]
    test_path = "landmark/tests/unit/infrastructure"
    fake_data = os.path.join(home, test_path, "fakeresults.csv")
    expected = pd.read_csv(fake_data, index_col="id")

    print(expected)
    print(truth)
    assert truth.equals(expected)
