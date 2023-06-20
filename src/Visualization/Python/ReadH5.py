# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from typing import Iterable, List, Tuple, Union

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def available_subfiles(h5file: h5py.File, extension: str) -> List[str]:
    """List all subfiles with the given 'extension' in the 'h5file'.

    Parameters
    ----------
    h5file: Open h5py file
    extension: str

    Returns
    -------
    List of paths in the 'h5file' that end with the 'extension'
    """
    subfiles = []

    def visitor(name):
        if name.endswith(extension):
            subfiles.append(name)

    h5file.visit(visitor)
    return subfiles


def to_dataframe(
    open_subfile: Union[h5py.Dataset, "spectre.IO.H5.H5Dat"], slice=None
) -> "pandas.DataFrame":
    """Convert a '.dat' subfile to a Pandas DataFrame

    This function isn't particularly complex, but it allows to convert a
    subfile to a DataFrame in a single statement like this:

        to_dataframe(open_h5_file["Norms.dat"])

    Without this function, you would have to store the subfile in an extra
    variable to access its "Legend" attribute.

    You can optionally pass a slice object which will slice the data for you
    so the entire dataset isn't read in

    Arguments:
      open_subfile: An open h5py subfile representing a SpECTRE dat file,
        or a spectre.IO.H5.H5Dat subfile, typically from a reductions file.
      slice: A numpy slice object to choose specific rows. Defaults to None. If
        you try to slice columns, an error will occur.

    Returns: Pandas DataFrame with column names read from the "Legend"
      attribute of the dat file.
    """
    import pandas as pd

    try:
        # SpECTRE H5 dat subfile
        data = np.asarray(open_subfile.get_data())
        legend = open_subfile.get_legend()
    except AttributeError:
        # h5py subfile
        data = open_subfile
        legend = open_subfile.attrs["Legend"]

    if slice:
        data = data[slice]

    return pd.DataFrame(data, columns=legend)


def select_observation(
    volfiles: Union["spectre.IO.H5.H5Vol", Iterable["spectre.IO.H5.H5Vol"]],
    step: int = None,
    time: float = None,
) -> Tuple[int, float]:
    """Select an observation in the 'volfiles'

    Arguments:
      volfiles: Open spectre H5 volume files. Can be a single volfile or a list,
        but can also be an iterator that opens and closes the files on demand.
        The volfiles are assumed to be ordered in time, meaning that later
        volfiles contain the same or later observation IDs than earlier
        volfiles. This assumption should hold for volfiles from multiple nodes
        in segments like 'Segment*/VolumeData*.h5' and exists to avoid opening
        all volfiles from all segments. See 'step' and 'time' below for details.
      step: Select the observation with this step number, counting unique
        observation IDs from the start of the first volfile.
        Mutually exclusive with 'time'.
      time: Select the observation closest to this time. The search is stopped
        once a volfile's closest time is further away than the previous.

    Returns: Tuple of (observation ID, time).
    """
    assert (step is None) != (
        time is None
    ), "Specify either 'step' or 'time', but not both."
    try:
        iter(volfiles)
    except TypeError:
        volfiles = [volfiles]

    if step is not None:
        # Select the specified step
        all_obs_ids = set()
        for volfile in volfiles:
            obs_ids = volfile.list_observation_ids()
            idx = step - len(all_obs_ids)
            if idx < len(obs_ids):
                obs_id = obs_ids[idx]
                obs_value = volfile.get_observation_value(obs_id)
                logger.info(
                    f"Selected observation step {step} at t = {obs_value:g}."
                )
                return obs_id, obs_value
            all_obs_ids.update(obs_ids)
        raise ValueError(
            f"Number of observations ({len(all_obs_ids)}) is smaller than"
            f" specified 'step' ({step})."
        )
    else:
        # Find closest observation to the specified time
        min_time_diff = np.inf
        min_step, obs_id, obs_value = None, None, None
        all_obs_ids = set()
        for volfile in volfiles:
            obs_ids = volfile.list_observation_ids()
            obs_values = list(map(volfile.get_observation_value, obs_ids))
            time_diff = np.abs(time - np.asarray(obs_values))
            step = np.argmin(time_diff)
            if time_diff[step] > min_time_diff:
                # Times in this volfile are further away from the target than
                # times in a previous volfile. Stop and return, since we assume
                # that volfiles are ordered.
                break
            min_time_diff = time_diff[step]
            min_step = len(all_obs_ids) + step
            obs_id, obs_value = obs_ids[step], obs_values[step]
            all_obs_ids.update(obs_ids)
        if obs_value != time:
            logger.info(
                f"Selected closest observation to t = {time}: "
                f"step {min_step} at t = {obs_value:g}"
            )
        return obs_id, obs_value
