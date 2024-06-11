"""
_summary_
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from behavysis_core.constants import IndivColumns
from behavysis_core.data_models.experiment_configs import ExperimentConfigs
from behavysis_core.mixins.df_io_mixin import DFIOMixin
from behavysis_core.mixins.keypoints_mixin import KeypointsMixin


class KeypointsModel:
    """
    _summary_
    """

    raw_dlc_df: pd.DataFrame
    annot_dlc_df: pd.DataFrame
    indivs_bpts_ls: pd.MultiIndex
    colours_ls = np.ndarray
    pcutoff: float
    radius: int
    colour_level: str
    cmap: str

    def __init__(self):
        self.raw_dlc_df = None
        self.annot_dlc_df = None
        self.indivs_bpts_ls = None
        self.colours_ls = None
        self.pcutoff = None
        self.radius = None
        self.colour_level = None
        self.cmap = None

    def load(self, fp: str, configs: ExperimentConfigs):
        """
        load in the raw DLC dataframe and set the configurations, from
        the given dlc_fp and configs.

        Parameters
        ----------
        fp : str
            _description_
        configs : ExperimentConfigs
            _description_
        """
        self.raw_dlc_df = KeypointsMixin.read_feather(fp)
        self.set_configs(configs)
        self.dlc_2_annot()

    def set_configs(self, configs: ExperimentConfigs):
        """
        Sets the configurations for the keypoints model.

        Parameters
        ----------
        configs : ExperimentConfigs
            The experiment configurations.
        """
        configs_filt = configs.user.evaluate.eval_vid
        self.colour_level = configs.get_ref(configs_filt.colour_level)
        self.pcutoff = configs.get_ref(configs_filt.pcutoff)
        self.radius = configs.get_ref(configs_filt.radius)
        self.cmap = configs.get_ref(configs_filt.cmap)

    def dlc_2_annot(self):
        """
        Converts the raw DLC dataframe to a row and column format that is ready for cv2
        video annotation.

        This method performs several operations on the raw DLC dataframe to prepare it for
        annotation. It cleans the headings, filters out specific columns, rounds and casts
        the values to the correct dtypes, changes the columns MultiIndex to a single-level
        index, creates a corresponding colours list for each bodypart instance,
        and saves the modified data to instance variables.

        Returns:
            None
        """
        dlc_df = self.raw_dlc_df.copy()
        dlc_df = KeypointsMixin.clean_headings(dlc_df)
        # Modifying dlc_df and making list of how to select dlc_df components to optimise processing
        # Filtering out PROCESS "invidividuals" in columns
        if IndivColumns.PROCESS.value in dlc_df.columns.unique("individuals"):
            dlc_df.drop(columns=IndivColumns.PROCESS.value, level="individuals")
        # Getting (indivs, bpts) MultiIndex
        indivs_bpts_ls = dlc_df.columns.droplevel("coords").unique()
        # Rounding and casting to correct dtypes - "x" and "y" values are ints
        dlc_df = dlc_df.fillna(0)
        columns = dlc_df.columns[
            dlc_df.columns.get_level_values("coords").isin(["x", "y"])
        ]
        dlc_df[columns] = dlc_df[columns].round(0).astype(int)
        # Changing the columns MultiIndex to a single-level index. For speedup
        dlc_df.columns = [
            f"{indiv}_{bpt}_{coord}" for indiv, bpt, coord in dlc_df.columns
        ]
        # Making the corresponding colours list for each bodypart instance
        colours_i, _ = pd.factorize(indivs_bpts_ls.get_level_values(self.colour_level))
        colours_ls = plt.get_cmap(self.cmap)(colours_i / colours_i.max())
        colours_ls = colours_ls[:, [2, 1, 0, 3]] * 255
        # Saving data to instance
        self.annot_dlc_df = dlc_df
        self.indivs_bpts_ls = indivs_bpts_ls
        self.colours_ls = colours_ls

    def annot_keypoints(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        """
        Adding the keypoints (given in frame number) to the frame and returning the annotated frame.

        Parameters
        ----------
        frame : np.ndarray
            cv2 frame array.
        frame_num : int
            index (i.e. frame number) in DLC dataframe.

        Returns
        -------
        np.ndarray
            cv2 frame array.
        """
        # Checking that frame_num is in the DLC dataframe
        if frame_num not in self.annot_dlc_df.index:
            return frame
        # Getting the corresponding row in the DLC dataframe
        row = self.annot_dlc_df.loc[frame_num]
        # For each indiv-bpt, if likelihood is above pcutoff, draw the keypoint
        for i, (indiv, bpt) in enumerate(self.indivs_bpts_ls):
            if row[f"{indiv}_{bpt}_likelihood"] >= self.pcutoff:
                cv2.circle(
                    frame,
                    (int(row[f"{indiv}_{bpt}_x"]), int(row[f"{indiv}_{bpt}_y"])),
                    radius=self.radius,
                    color=self.colours_ls[i],
                    thickness=-1,
                )
        return frame
