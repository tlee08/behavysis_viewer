import numpy as np
from behavysis_core.data_models.bouts import Bouts
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.behav_mixin import BehavMixin
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from PySide6.QtCore import QAbstractListModel, Qt
from PySide6.QtGui import QColor

from behavysis_viewer.utils.constants import VALUE2COLOR


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
        # Loading behaviour data
        behavs_df = BehavMixin.read_feather(fp)
        # behavs_df to bouts
        self.bouts = BehavMixin.frames_2_bouts(behavs_df)
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
