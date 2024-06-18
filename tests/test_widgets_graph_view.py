import numpy as np
import pandas as pd
import pytest
from behavysis_core.constants import BehavCN
from behavysis_core.mixins.behav_mixin import BehavMixin
from behavysis_core.mixins.df_io_mixin import DFIOMixin

from behavysis_viewer.widgets.graph_view import GraphView


@pytest.fixture(scope="module", autouse=True)
def graph_view(app):
    # Create an instance of the CvView widget
    graph_view = GraphView()
    graph_view.setFixedSize(400, 300)

    yield graph_view

    # Clean up the CvView widget
    graph_view.close()


@pytest.fixture(scope="module", autouse=False)
def rand_bouts():
    # Making random behavs frames df
    frames_vect = np.arange(0, 10000)
    pred_vect = np.sin(frames_vect) + (np.random.rand(10000) * 2 - 1)
    pred_vect = pred_vect > 0
    frames_df = pd.DataFrame({("my_behav", "pred"): pred_vect}, index=frames_vect)
    frames_df.columns = pd.MultiIndex.from_tuples(
        frames_df.columns, names=DFIOMixin.enum_to_list(BehavCN)
    )
    # Adding in "actual" column  and "my_user_behav" column
    frames_df = BehavMixin.frames_add_behaviour(frames_df, ["my_user_behav"])
    # frames_df to bouts
    bouts = BehavMixin.frames_to_bouts(frames_df)

    yield bouts


@pytest.fixture(scope="module", autouse=True)
def plot_init_bouts(graph_view, rand_bouts):
    # Updating graph_viewer with bouts data
    start_ls = np.array([i.start for i in rand_bouts.bouts]) / 15
    stop_ls = np.array([i.stop for i in rand_bouts.bouts]) / 15
    behavs_ls = np.array([i.behaviour for i in rand_bouts.bouts])
    graph_view.plot_init(start_ls, stop_ls, behavs_ls)


def test_graph_view_initialization(graph_view):
    # Test that the CvView widget is initialized correctly
    assert graph_view is not None


def test_graph_view_plot_init(graph_view):
    assert graph_view is not None


def test_graph_view_plot_update(graph_view):
    graph_view.plot_update(15, xmin=10, xmax=20)
    assert graph_view is not None


def test_graph_view_plot_2_cv(graph_view):
    img_cv = graph_view.plot_2_cv()
    assert isinstance(img_cv, np.ndarray)
    assert img_cv.shape == (graph_view.height(), graph_view.width(), 3)


if __name__ == "__main__":
    pytest.main([__file__])
