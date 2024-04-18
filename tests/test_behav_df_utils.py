import numpy as np
import pandas as pd
import pytest

from ba_viewer.utils import behavs_df_utils


@pytest.fixture(scope="function", autouse=False)
def behavs_df():
    df = pd.read_feather("tests/resources/behavs_df.feather")
    df = df.reindex(columns=sorted(df.columns))
    yield df


@pytest.fixture(scope="function", autouse=False)
def rand_img():
    return np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)


def test_frames_2_bouts_2_frames(behavs_df):
    bouts = behavs_df_utils.frames_to_bouts(behavs_df)
    assert bouts.start == behavs_df.index[0]
    assert bouts.stop == behavs_df.index[-1] + 1
    assert bouts.bouts is not None

    behavs_df2 = behavs_df_utils.bouts_2_frames(bouts)
    assert behavs_df2.equals(behavs_df)

    bouts2 = behavs_df_utils.frames_to_bouts(behavs_df2)
    assert bouts == bouts2
