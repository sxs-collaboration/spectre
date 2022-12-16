# Distributed under the MIT License.
# See LICENSE.txt for details.

import h5py


def available_subfiles(h5file, extension):
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
