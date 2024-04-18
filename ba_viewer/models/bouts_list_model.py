import numpy as np
from PySide6.QtCore import QAbstractListModel, Qt
from PySide6.QtGui import QColor

from ba_viewer.pydantic_models.bouts import Bouts
from ba_viewer.utils.constants import VALUE2COLOR

from ba_viewer.pydantic_models.experiment_configs import ExperimentConfigs

from ba_viewer.utils import behavs_df_utils
from ba_viewer.utils.funcs import read_feather


class BoutsListModel(QAbstractListModel):

    bouts: Bouts

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bouts = Bouts(start=-1, stop=-1, bouts=[])

    def data(self, index, role):
        """
        Displays data in QListView.

        This is a required function.
        """
        bout = self.bouts.bouts[index.row()]
        # Displays text
        if role == Qt.ItemDataRole.DisplayRole:
            return f"{bout.behaviour} - {index.row()}"
        # Displays background colour
        if role == Qt.ItemDataRole.BackgroundRole:
            return QColor(VALUE2COLOR[bout.actual])

    def setData(self, index, value, role):
        # Updates checkbox
        # if role == Qt.ItemDataRole.CheckStateRole:
        #     self.bouts_df[index.row()][3] = value
        # self.dataChanged.emit(index, index)
        return True

    def load(self, fp: str, configs: ExperimentConfigs):
        # Getting necessary configs
        user_behavs = configs.user.classify_behaviours.user_behavs
        # Loading data
        behavs_df = read_feather(fp)
        # Adding actual and user_behavs to behavs_df (if they aren't already there)
        behavs_df = behavs_df_utils.frames_add_behaviour(behavs_df, user_behavs)
        # behavs_df to bouts
        self.bouts = behavs_df_utils.frames_to_bouts(behavs_df)
        # print(self.bouts.model_dumps_json(indent=2))
        self.layoutChanged.emit()

    def rowCount(self, index):
        """
        Gets number of rows for QListView.

        This is a required function.
        """
        return len(self.bouts.bouts)

    def flags(self, index):
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsUserCheckable
        )


def serialiser(i):
    if isinstance(i, np.integer):
        return int(i)
    else:
        return i
