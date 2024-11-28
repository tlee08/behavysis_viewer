import numpy as np
import pytest
from behavysis_core.df_classes.bouts_df import BoutsDf


@pytest.fixture(scope="function", autouse=False)
def behavs_df():
    df = BoutsDf.read_feather("tests/resources/behavs_df.feather")
    df = df.reindex(columns=sorted(df.columns))
    yield df


@pytest.fixture(scope="function", autouse=False)
def rand_img():
    return np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)


def test_frames2bouts2frames(behavs_df):
    bouts = BoutsDf.frames2bouts(behavs_df)
    assert bouts.start == behavs_df.index[0]
    assert bouts.stop == behavs_df.index[-1] + 1
    assert bouts.bouts is not None

    behavs_df2 = BoutsDf.bouts2frames(bouts)
    assert behavs_df2.equals(behavs_df)

    bouts2 = BoutsDf.frames2bouts(behavs_df2)
    assert bouts == bouts2
