import sys
import time
from multiprocessing import Process, Value
from multiprocessing.sharedctypes import Synchronized

import cv2
import numpy as np
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.behaviour_mixin import BehaviourMixin
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
)
from tqdm import trange

from behavysis_viewer.models.bout_inspect_list_model import BoutInspectListModel
from behavysis_viewer.models.bouts_list_model import BoutsListModel
from behavysis_viewer.models.exp_file_manager import ExpFileManager
from behavysis_viewer.models.keypoints_model import KeypointsModel
from behavysis_viewer.models.vid_model import VidModel
from behavysis_viewer.ui.main_ui import Ui_MainWindow
from behavysis_viewer.utils.constants import STATUS_MSG_TIMEOUT
from behavysis_viewer.widgets.graph_view import GraphView
from behavysis_viewer.windows.help import HelpWindow
from behavysis_viewer.windows.settings import SettingsWindow
from behavysis_viewer.windows.window_mixin import WindowMixin


class MainWindow(QMainWindow, WindowMixin):
    """__summary__"""

    ui: Ui_MainWindow
    preferences_window: SettingsWindow
    help_window: HelpWindow

    file_manager: ExpFileManager
    vid_model: VidModel
    keypoints_model: KeypointsModel

    window_size: int
    curr_i: int

    bouts_model: BoutsListModel
    bout_inspect_model: BoutInspectListModel

    def __init__(self, *args, **kwargs):
        # Instatiating QMainWindow
        super().__init__(*args, **kwargs)

        # Connecting main window with UI (from Qt-Designer)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Initialising settings window widget
        self.settings_window = SettingsWindow(main=self)
        # Initialising help window widget
        self.help_window = HelpWindow(main=self)

        # Initialising vid graph model
        self._init_models()

        # Grouping radio-buttons together in dict
        self.rbtns = {
            1: self.ui.is_behav_rbtn,
            0: self.ui.not_behav_rbtn,
            -1: self.ui.select_behav_rbtn,
        }

        self.resize_viewer(360, 240)

        # Initialising models and connecting to views
        self._init_models()
        self._init_model_views()

        # Connecting signals and slots
        self._init_conns_vid()
        self._init_conns_models()
        self._init_conns_views()
        self._init_conns_scoring()
        self._init_conns_io()
        self._init_conns_shortcuts()
        self._init_conns_hotkeys()

        # Initialising timer to play video
        self._init_timer_vid()

    def _init_models(self):
        """__summary__"""
        # Init file manager
        self.file_manager = ExpFileManager()
        # Init vid and keypoints models
        self.vid_model = VidModel()
        self.keypoints_model = KeypointsModel()
        # Init bouts and bout_inspect list models
        self.bouts_model = BoutsListModel()
        self.bout_inspect_model = BoutInspectListModel()
        # Init primitive variables
        self.window_size = 25
        self.curr_i = 0

    def _init_model_views(self):
        """__summary__"""
        # MODEL VIEWS
        # Linking BoutsModel
        self.ui.bouts_view.setModel(self.bouts_model)
        # Linking BoutInspectModel
        self.ui.bout_inspect_view.setModel(self.bout_inspect_model)

    def _init_conns_models(self):
        """__summary__"""
        # SIGNALS AND SLOTS: MODELS
        # bouts_model
        m = self.bouts_model
        # bout_inspect_model
        m = self.bout_inspect_model
        m.layoutChanged.connect(
            lambda: self.ui.bout_inspect_widget.setEnabled(m.is_selected)
        )
        m.layoutChanged.connect(
            lambda: self.ui.bout_inspect_view.setEnabled(m.bout.actual == 1)
        )

    def _init_conns_views(self):
        """__summary__"""
        # SIGNALS AND SLOTS: VIEWS
        # bouts_view
        v = self.ui.bouts_view.selectionModel()
        v.selectionChanged.connect(self.select_bout)
        # bout_inspect_view
        v = self.ui.bout_inspect_view.selectionModel()
        v.selectionChanged.connect(v.clearSelection)

    def _init_conns_scoring(self):
        """__summary__"""
        # SIGNALS AND SLOTS: SCORING BOUT
        # Choosing is_actual bout
        for k, v in self.rbtns.items():
            v.toggled.connect(lambda r, k=k: self.toggle_actual_rbtns(r, k))
        # Choosing specific behaviours
        # NOTE: already done in bout_inspect_model.setData, when CheckStateRole changes.

    def _init_conns_vid(self):
        """__summary__"""
        # SIGNALS AND SLOTS: VIDEO AND PLOT
        # Handle user moving slider
        self.ui.slider.sliderMoved.connect(self.set_frame)
        self.ui.slider.sliderPressed.connect(
            lambda: self.set_frame(self.ui.slider.value())
        )
        # Handle video buttons
        self.ui.start_stop_btn.toggled.connect(
            lambda i: self.timer.stop() if i else self.timer.start()
        )
        self.ui.vid_back_btn.clicked.connect(
            lambda: self.set_frame(self.curr_i - self.vid_model.jump_size)
        )
        self.ui.vid_fwd_btn.clicked.connect(
            lambda: self.set_frame(self.curr_i + self.vid_model.jump_size)
        )
        # Handle bout-related video buttons
        self.ui.bout_replay_btn.clicked.connect(
            lambda: self.set_frame(self.bout_inspect_model.start)
        )

    def _init_conns_io(self):
        """__summary__"""
        # SIGNALS AND SLOTS: I/O
        # Handle opening settings and help windows
        self.ui.action_settings.triggered.connect(
            lambda: self.toggle_window(self.settings_window)
        )
        self.ui.action_help.triggered.connect(
            lambda: self.toggle_window(self.help_window)
        )
        # Handle opening
        self.ui.action_open.triggered.connect(self.open)
        # Handle saving
        self.ui.action_save.triggered.connect(self.save)
        self.ui.action_save_as_frames.triggered.connect(self.save_frames)
        self.ui.action_save_as_bouts.triggered.connect(self.save_bouts)
        # Handle exporting
        self.ui.action_export_video.triggered.connect(self.export_vid)
        # Handle open/closing windows
        self.ui.action_quit.triggered.connect(self.close)

    def _init_conns_shortcuts(self):
        """__summary__"""
        # SLOTS AND SIGNALS: KEYBOARD SHORTCUTS
        self.ui.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.ui.action_save.setShortcut(QKeySequence.StandardKey.Save)
        self.ui.action_quit.setShortcut(QKeySequence.StandardKey.Close)
        self.ui.action_settings.setShortcut(Qt.Key.Key_P)
        self.ui.action_help.setShortcut(Qt.Key.Key_H)

    def _init_conns_hotkeys(self):
        """__summary__"""
        # SLOTS AND SIGNALS: HOTKEYS
        # Handle video hot-keys
        QShortcut(Qt.Key.Key_Left, self).activated.connect(self.ui.vid_back_btn.click)
        QShortcut(Qt.Key.Key_Right, self).activated.connect(self.ui.vid_fwd_btn.click)
        QShortcut(Qt.Key.Key_Space, self).activated.connect(
            self.ui.start_stop_btn.toggle
        )
        QShortcut(Qt.Key.Key_K, self).activated.connect(self.ui.annot_keypts_cbx.toggle)
        # Handle bout-related hot-keys
        QShortcut(Qt.Key.Key_R, self).activated.connect(self.ui.bout_replay_btn.click)
        QShortcut(Qt.Key.Key_F, self).activated.connect(self.ui.bout_focus_btn.toggle)
        QShortcut(Qt.Key.Key_1, self).activated.connect(self.ui.is_behav_rbtn.toggle)
        QShortcut(Qt.Key.Key_2, self).activated.connect(self.ui.not_behav_rbtn.toggle)
        QShortcut(Qt.Key.Key_3, self).activated.connect(
            self.ui.select_behav_rbtn.toggle
        )

    def _init_timer_vid(self):
        """__summary__"""
        # TIMERS AND THREADS
        # Making timer to play video
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def resize_viewer(self, width, height):
        """__summary__"""
        self.ui.vid_viewer.setFixedSize(width, height)
        self.ui.graph_viewer.setFixedWidth(width)
        self.ui.slider.setFixedWidth(width)

    def closeEvent(self, event):
        """__summary__"""
        # Perform actions when the window is about to close
        reply = QMessageBox.question(
            self,
            "Message",
            "Are you sure you want to close the window?",
            QMessageBox.StandardButton.Yes,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.settings_window.close()
            event.accept()
        else:
            event.ignore()

    def open(self, fp: str | None = None) -> None:
        """__summary__"""
        if not fp:
            fp = QFileDialog.getOpenFileName(
                self, "", "", "config file (*.json *.yaml)"
            )[0]
        # Getting corresponding experiment filenames
        if not fp:
            self.ui.statusbar.showMessage(
                "No video selected", timeout=STATUS_MSG_TIMEOUT
            )
            return
        try:
            # Loading filenames in vid file manager
            self.file_manager.load(fp)
            # Reading in configs
            configs = ExperimentConfigs.read_json(self.file_manager.configs_fp)
            # Loading data into vid, bouts, and keypoints models
            self.vid_model.load(self.file_manager.vid_fp)
            self.bouts_model.load(self.file_manager.behavs_df_fp, configs)
            self.keypoints_model.load(self.file_manager.dlc_df_fp, configs)
            # Setting window size
            self.curr_i = 0
            # Preparing slider
            self.ui.slider.setMaximum(self.vid_model.nframes)
            # Preparing QTimer to play video (at given fps)
            self.timer.start(1000 // self.vid_model.fps)
            # Updating the graph_viewer with bouts data
            self.ui.graph_viewer.plot_bouts_init(self.bouts_model.bouts, configs)
            # Writing msg to statusbar
            self.ui.statusbar.showMessage(
                f"Opened video: {fp}", timeout=STATUS_MSG_TIMEOUT
            )
        except ValueError as e:
            self.ui.statusbar.showMessage(
                f"Failed to open {fp}: {e}", timeout=STATUS_MSG_TIMEOUT
            )
            print(e)

    def select_bout(self):
        """__summary__"""
        # Getting selected bout QIndex (if there exists one)
        if len(self.ui.bouts_view.selectedIndexes()) > 0:
            index = self.ui.bouts_view.selectedIndexes()[0]
            # Getting bout df row
            bout = self.bouts_model.bouts.bouts[index.row()]
            # Loading in BoutInspectModel row
            self.bout_inspect_model.load(bout)
            # Setting bout inspect header text
            self.ui.bout_inspect_header.setText(f"{index.row()} - {bout.behaviour}")
            # Linking is_behav rbtns
            self.rbtns[bout.actual].toggle()
            # Jumping to bout start
            self.set_frame(bout.start)

    def update_frame(self):
        """__summary__"""
        # If bout_focus_btn is checked AND video is past the current bout's end,
        # then pause (i.e., don't update the frame)
        if self.ui.bout_focus_btn.isChecked():
            if self.curr_i > self.bout_inspect_model.stop:
                return
        # Update vid frame
        ret = self.update_frame_vid()
        if ret:
            # Update plot
            self.update_frame_plot()
            # Update slider
            self.ui.slider.setValue(self.curr_i)
            # Update curr_frame value
            self.curr_i += 1

    def update_frame_vid(self):
        """__summary__"""
        # Reading in next frame
        ret, frame = self.vid_model.read()
        if ret:
            # Checking whether to annotate keypoints
            if self.ui.annot_keypts_cbx.isChecked():
                # Updating frame with annotated keypoints
                frame = self.keypoints_model.annot_keypoints(frame, self.curr_i)
            # Displaying frame on cv_viewer
            self.ui.vid_viewer.display_cv2(frame)
        return ret

    def update_frame_plot(self):
        """__summary__"""
        self.ui.graph_viewer.plot_update(
            self.curr_i / self.vid_model.fps,
            xmin=(self.curr_i - self.window_size) / self.vid_model.fps,
            xmax=(self.curr_i + self.window_size) / self.vid_model.fps,
        )

    def set_frame(self, frame_num):
        """__summary__"""
        # Setting curr_frame value
        self.curr_i = np.clip(frame_num, 0, self.vid_model.nframes - 1)
        # Setting video to new frame
        self.vid_model.vid.set(cv2.CAP_PROP_POS_FRAMES, self.curr_i)
        # Updating video and plot view
        self.update_frame()

    def toggle_actual_rbtns(self, r: bool, value: int):
        """__summary__"""
        if r:
            self.bout_inspect_model.actual = value

    def save(self):
        """__summary__"""
        self.save_frames(self.file_manager.behavs_df_fp)

    def save_frames(self, fp=None):
        """__summary__"""
        if not fp:
            fp = QFileDialog.getSaveFileName(
                self,
                "",
                f"{self.file_manager.name}.feather",
                "feather dataframe (*.feather)",
            )[0]
        if fp:
            # bouts to behavs_df
            behavs_df = BehaviourMixin.bouts_2_frames(self.bouts_model.bouts)
            # Writing to feather file
            DFIOMixin.write_feather(behavs_df, fp)
            self.ui.statusbar.showMessage(
                f"Saved scored behaviour frames to {fp}", timeout=STATUS_MSG_TIMEOUT
            )

    def save_bouts(self, fp=None):
        """__summary__"""
        if not fp:
            fp = QFileDialog.getSaveFileName(
                self,
                "",
                f"{self.file_manager.name}.json",
                "json file (*.json *.yaml)",
            )[0]
        if fp:
            # Writing to json file
            self.bouts_model.bouts.write_json(fp)
            self.ui.statusbar.showMessage(
                f"Saved scored behaviour bouts to {fp}", timeout=STATUS_MSG_TIMEOUT
            )

    def export_vid(self, fp=None):
        """__summary__"""
        if not fp:
            fp = QFileDialog.getSaveFileName(
                self,
                "",
                f"{self.file_manager.name}.mp4",
                "video (*.mp4)",
            )[0]
        if fp:
            # Annotating each frame using the created functions
            i = Value("I", 0, lock=True)
            p = Process(
                target=self.export_vid_worker,
                args=(
                    self.file_manager,
                    fp,
                    self.ui.vid_viewer.width(),
                    self.ui.vid_viewer.height(),
                    self.window_size,
                    i,
                ),
            )

            p.start()

            # while i.value < self.vid_model.nframes:
            #     time.sleep(0.5)
            #     with i.get_lock():
            #         self.ui.statusbar.showMessage(
            #             f"Exporting video: {i}/{self.vid_model.nframes}",
            #             timeout=STATUS_MSG_TIMEOUT,
            #         )

            p.join()

            # Displaying message
            self.ui.statusbar.showMessage(
                f"Exported video to {fp}", timeout=STATUS_MSG_TIMEOUT
            )

    @staticmethod
    def export_vid_worker(
        file_manager: ExpFileManager,
        out_fp: str,
        w: int,
        h: int,
        window_size: int,
        i: Synchronized,
    ):
        """Annotating video with keypoints and scored behaviours loop."""
        # Define the codec and create VideoWriter object
        app = QApplication()

        configs = ExperimentConfigs.read_json(file_manager.configs_fp)

        vid_model = VidModel()
        vid_model.load(file_manager.vid_fp)

        bouts_model = BoutsListModel()
        bouts_model.load(file_manager.behavs_df_fp, configs)

        graph_viewer = GraphView()
        graph_viewer.plot_bouts_init(bouts_model.bouts, configs)
        graph_viewer.setFixedSize(w, h)

        keypoints_model = KeypointsModel()
        keypoints_model.load(file_manager.dlc_df_fp, configs)

        out_cap = cv2.VideoWriter(
            out_fp,
            cv2.VideoWriter_fourcc(*"mp4v"),
            vid_model.fps,
            (w, h * 2),
        )

        # Annotating each frame using the created functions
        for _ in trange(vid_model.nframes):
            # for _ in np.arange(vid_model.nframes):
            # Reading in next frame
            ret, frame = vid_model.read()
            if ret is False:
                break
            frame = keypoints_model.annot_keypoints(frame, i.value)
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            # Making graph frame
            graph_viewer.plot_update(
                i.value / vid_model.fps,
                xmin=(i.value - window_size) / vid_model.fps,
                xmax=(i.value + window_size) / vid_model.fps,
            )
            graph_frame = graph_viewer.plot_2_cv()
            # Writing annotated frame to the VideoWriter
            out_cap.write(np.concatenate((frame, graph_frame), axis=0))
            # Update i
            with i.get_lock():
                i.value += 1
        # Release the VideoWriter (i.e. save)
        out_cap.release()
        # Close the graph_viewer
        app.quit()


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
    sys.exit(app.exec())
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
    sys.exit(app.exec())
