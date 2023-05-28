# Distributed under the MIT License.
# See LICENSE.txt for details.

from typing import List, Union

import h5py
import numpy as np


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
