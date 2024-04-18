"""
Utility functions.
"""

import os
import re
from inspect import currentframe
from typing import Callable, TypeVar, Type

from pydantic import BaseModel

import cv2
import numpy as np
import pandas as pd

from ba_viewer.utils.constants import (
    DIAGNOSTICS_SUCCESS_MESSAGES,
    DLC_COLUMN_NAMES,
    HDF_KEY,
    PROCESS_COL,
    SINGLE_COL,
)

#####################################################################
#               CONFIG FILE HELPER FUNCS
#####################################################################

T = TypeVar("T", bound=BaseModel)


def read_configs(fp: str, model_class: Type[T]) -> T:
    """
    Returns the config model from the specified JSON config file.

    Parameters
    ----------
    fp : str
        Filepath of the JSON config file.
    model_class : Type[T]
        The BaseModel class for type hints.

    Returns
    -------
    T
        The config model.

    Notes
    -----
    This function reads the contents of the JSON config file located at `fp` and
    returns the config model. The `model_class` parameter is used for type hints.
    The `T` type parameter should extend `BaseModel`.

    Example
    -------
    >>> config = read_configs("/path/to/config.json", ConfigModel)
    """
    with open(fp, "r", encoding="utf-8") as f:
        return model_class.model_validate_json(f.read())


def write_configs(configs: BaseModel, fp: str) -> None:
    """
    Writes the given configs model to the configs file (i.e. hence updating the file)

    Parameters
    ----------
    configs : Configs
        Configs model to write to file.
    fp : str
        File to save configs to.
    """
    os.makedirs(os.path.split(fp)[0], exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        f.write(configs.model_dump_json(indent=2))


#####################################################################
#               DATA FRAME READER/WRITER FUNCS (CSV, H5, Feather)
#####################################################################


def read_dlc_csv(fp: str) -> pd.DataFrame:
    """
    Reading in DLC csv file.

    Parameters
    ----------
    fp : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    try:
        return pd.read_csv(
            fp, header=np.arange(len(DLC_COLUMN_NAMES)).tolist(), index_col=0
        ).sort_index()
    except Exception as e:
        raise ValueError(
            f'The csv file, "{fp}", does not exist or is in an invalid format.'
            + "Please check this file."
        ) from e


def write_dlc_csv(df: pd.DataFrame, fp: str) -> None:
    """
    Writing DLC dataframe to csv file.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    fp : str
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    os.makedirs(os.path.split(fp)[0], exist_ok=True)
    try:
        df.to_csv(fp)
    except Exception as e:
        raise ValueError(e) from e


def read_h5(fp: str) -> pd.DataFrame:
    """
    Reading h5 file.

    Parameters
    ----------
    fp : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    try:
        return pd.DataFrame(pd.read_hdf(fp, key=HDF_KEY, mode="r").sort_index())
    except Exception as e:
        raise ValueError(
            f'The h5 file, "{fp}", does not exist or is in an invalid format.'
            + "Please check this file."
        ) from e


def write_h5(df: pd.DataFrame, fp: str) -> None:
    """
    Writing dataframe h5 file.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    fp : str
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    os.makedirs(os.path.split(fp)[0], exist_ok=True)
    try:
        df.to_hdf(fp, key=HDF_KEY, mode="w")
    except Exception as e:
        raise ValueError(e) from e


def read_feather(fp: str) -> pd.DataFrame:
    """
    Reading feather file.

    Parameters
    ----------
    fp : str
        _description_

    Returns
    -------
    pd.DataFrame
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    try:
        return pd.read_feather(fp).sort_index()
    except Exception as e:
        raise ValueError(
            f'The feather file, "{fp}", does not exist or is in an invalid format.'
            + "Please check this file."
        ) from e


def write_feather(df: pd.DataFrame, fp: str) -> None:
    """
    Writing dataframe feather file.

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    fp : str
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    os.makedirs(os.path.split(fp)[0], exist_ok=True)
    try:
        df.to_feather(fp)
    except Exception as e:
        raise ValueError(e) from e


#####################################################################
#               PARSING VIDEO METADATA
#####################################################################


def read_video_metadata(fp: str) -> dict[str, float]:
    """
    Finds the experiment's video metadata/parameters for either the raw or formatted video,
    and stores this data in the experiment's config file. This includes:

    - `height_px`: resolution given by number of pixels height-wise.
    - `width_px`: resolution given by number of pixels width-wise.
    - `total_frames`: the toal number of frames in the video.
    - `fps`: frames per second.
    - `dur_sec`: duration of the video in seconds.


    Parameters
    ----------
    fp : str
        fp: The video filepath to retrieve the metadata from.

    Returns
    -------
    dict[str, float]
        Description of the function's outcome

    Raises
    ------
    ValueError
        _description_
    """
    meta_dict = {}
    # Reading in video metadata
    cap = cv2.VideoCapture(fp)
    # Assertion: video file should be valid
    if not cap.isOpened():
        raise ValueError(
            f"The file, {fp}, does not exist or is corrupted. Please check this file."
        )
    meta_dict["height_px"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta_dict["width_px"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    meta_dict["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta_dict["fps"] = cap.get(cv2.CAP_PROP_FPS)
    meta_dict["dur_sec"] = meta_dict["total_frames"] / meta_dict["fps"]
    cap.release()
    return meta_dict


#####################################################################
#               MISC FUNCS
#####################################################################


def success_msg() -> str:
    """
    Return a random positive message :)

    Returns
    -------
    str
        _description_
    """
    return np.random.choice(DIAGNOSTICS_SUCCESS_MESSAGES)


def check_bpts_exist(bodyparts: list, dlc_df: pd.DataFrame) -> None:
    """
    _summary_

    Parameters
    ----------
    bodyparts : list
        _description_
    dlc_df : pd.DataFrame
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    # Checking that the bodyparts are all valid:
    bodyparts_exist = np.isin(bodyparts, dlc_df.columns.unique("bodyparts"))
    if not bodyparts_exist.all():
        msg = (
            "Some bodyparts in the config file are missing from the csv file.\n"
            + "They are:\n"
        )
        for bp in np.array(bodyparts)[~bodyparts_exist]:
            msg += f"    - {bp}\n"
        raise ValueError(msg)


def warn_overwrite_msg(func: Callable = None):
    """
    Return a warning message for the given function.

    Parameters
    ----------
    func : Callable, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    if not func:
        func = currentframe().f_back.f_code.co_name
    return (
        "WARNING: Output file already exists - not overwriting file.\n"
        + "To overwrite, specify {}(..., overwrite=True)."
    ).format(func)


def get_dlc_headings(dlc_df: pd.DataFrame) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Returns a tuple of the individuals (animals, not "single"), and a tuple of the multi-animal
    bodyparts.

    Parameters
    ----------
    dlc_df : pd.DataFrame
        DLC pd.DataFrame.

    Returns
    -------
    tuple[tuple[str, ...], tuple[str, ...]]
        `(indivs_ls, bpts_ls)` tuples. It is recommended to unpack these vals.
    """
    # Getting DLC column MultiIndex
    columns = dlc_df.columns
    # Filtering out any single and processing columns
    # Getting individuals to filter out
    filt_cols = [PROCESS_COL, SINGLE_COL]
    # Filtering out
    for filt_col in filt_cols:
        if filt_col in columns.unique("individuals"):
            columns = columns.drop(filt_col, level="individuals")
    # Getting individuals list
    indivs = columns.unique("individuals").to_list()
    # Getting bodyparts list
    bpts = columns.unique("bodyparts").to_list()
    return indivs, bpts


def clean_dlc_headings(dlc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops the "scorer" level (and any other unnecessary levels) in the column
    header of the dataframe. This makes analysis easier.

    Parameters
    ----------
    dlc_df : pd.DataFrame
        DLC pd.DataFrame.

    Returns
    -------
    pd.DataFrame
        DLC pd.DataFrame.
    """
    dlc_df = dlc_df.copy()
    # Removing the scorer column because all values are identical
    dlc_df.columns = dlc_df.columns.droplevel("scorer")
    # Grouping the columns by the individuals level for cleaner presentation
    dlc_df = dlc_df.reindex(
        columns=dlc_df.columns.unique("individuals"), level="individuals"
    )
    return dlc_df


#################################################
#               DIR HELPER FUNCS
#################################################


def clear_dir_junk(my_dir: str) -> None:
    """
    Removes all hidden files in given directory.
    Hidden files begin with ".".

    Parameters
    ----------
    my_dir : str
        Directory to clear.
    """
    for i in os.listdir(dir):
        path = os.path.join(my_dir, i)
        if re.search(r"^\.", i):
            os.remove(path)


def silent_remove(fp: str) -> None:
    """
    Removes the given file if it exists.
    Does nothing if not.
    Does not throw any errors,

    Parameters
    ----------
    fp : str
        Filepath to remove.
    """
    try:
        os.remove(fp)
    except OSError:
        pass


def get_name(fp: str) -> str:
    """
    Given the filepath, returns the name of the file.
    The name is:
    ```
    <path_to_file>/<name>.<ext>
    ```

    Parameters
    ----------
    fp : str
        Filepath.

    Returns
    -------
    str
        File name.
    """
    return os.path.splitext(os.path.split(fp)[1])[0]
