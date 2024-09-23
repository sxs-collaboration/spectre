#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from typing import Optional

import click
import h5py
import numpy as np
import rich
import scipy.spatial as spatial

from spectre.support.CliExceptions import RequiredChoiceError
from spectre.support.Logging import configure_logging
from spectre.Visualization.GenerateXdmf import available_subfiles

logger = logging.getLogger(__name__)


def generate_tetrahedral_connectivity(
    h5file,
    subfile_name: str,
    relative_paths: bool = True,
    start_time: Optional[float] = None,
    stop_time: Optional[float] = None,
    stride: int = 1,
    coordinates: str = "InertialCoordinates",
    force: bool = False,
):
    """Generate tetrahedral connectivity using scipy.spatial.Delaunay

    Given the coordinates, this generates a tetrahedral connectivity that can
    be read in by ParaView or other visualization software (e.g. VisIt or yt).
    It uses the scipy.spatial.Delaunay class to generate the connectivity, which
    uses the Qhull library (github.com/qhull/qhull/), which uses the quickhull
    algorithm and scales as O(n^2) in the worst case. Thus, generating the
    connectivity can take several minutes per temporal ID. Unfortunately,
    depending on the particular grid point distribution, generating the
    connectivity may even fail in qhull, which is unfortunately difficult
    to fix.

    You should combine the volume HDF5 files into one before running this in
    order to get a fully connected domain.

    Note that while ParaView has a Delaunay3D filter, it is much slower than
    qhull, needs to be rerun every time ParaView is opened while the qhull
    output is stored, and the Delaunay3D filter sometimes produces nonsense
    connectivity that is very difficult to debug and fix.

    After the tetrahedral connectivity has been written you can run
    'generate-xdmf' with the flag '--use-tetrahedral-connectivity'. ParaView
    volume renderings are sometimes nicer with tetrahedral connectivity and this
    can be used to fill gaps between finite-difference or Gauss elements.

    If this algorithm is too slow, one possible improvement is to apply
    qhull to each block of the domain and then connect the blocks to each
    other separately. This keeps the number of grid points lower for each
    invocation of qhull, which likely reduces total runtime and may also
    reduce or eliminate failure cases. In the ideal case we would apply
    qhull to each element and then connect elements that are using FD or
    Gauss points to their neighbors.

    \f
    Arguments:
      h5file: The HDF5 file on which to run.
      subfile_name: Volume data subfile in the H5 files.
      start_time: Optional. The earliest time at which to start visualizing. The
        start-time value is included.
      stop_time: Optional. The time at which to stop visualizing. The stop-time
        value is not included.
      stride: Optional. View only every stride'th time step.
      coordinates: Optional. Name of coordinates dataset. Default:
        "InertialCoordinates".
      force: Optional. Overwrite the existing tetrahedral connectivity.
        Default: False
    """
    filename = h5file
    h5file = h5py.File(filename, "a")

    if not subfile_name:
        subfiles = available_subfiles(h5file, extension=".vol")
        if len(subfiles) == 1:
            subfile_name = subfiles[0]
            logger.info(
                f"Selected subfile {subfile_name} (the only available one)."
            )
        else:
            raise RequiredChoiceError(
                (
                    "Specify '--subfile-name' / '-d' to select a"
                    " subfile containing volume data."
                ),
                choices=subfiles,
            )

    if not subfile_name.endswith(".vol"):
        subfile_name += ".vol"

    # Open subfile
    try:
        vol_subfile = h5file[subfile_name]
    except KeyError as err:
        raise ValueError(
            f"Could not open subfile name '{subfile_name}' in"
            f" '{filename}'. Available subfiles: "
            + str(available_subfiles(h5file, extension=".vol"))
        ) from err

    # Sort timesteps by time
    temporal_ids_and_values = sorted(
        [
            (key, vol_subfile[key].attrs["observation_value"])
            for key in vol_subfile.keys()
        ],
        key=lambda key_and_time: key_and_time[1],
    )

    # Stride through timesteps
    for temporal_id, time in temporal_ids_and_values[::stride]:
        # Filter by start and end time
        if start_time is not None and time < start_time:
            continue
        if stop_time is not None and time > stop_time:
            break

        data_at_id = vol_subfile[temporal_id]

        if force and ("tetrahedral_connectivity" in data_at_id):
            del data_at_id["tetrahedral_connectivity"]

        x_coords = np.asarray(data_at_id[coordinates + "_x"])
        y_coords = np.asarray(data_at_id[coordinates + "_y"])
        if (coordinates + "_z") in data_at_id:
            z_coords = np.asarray(data_at_id[coordinates + "_z"])
            coords = np.column_stack((x_coords, y_coords, z_coords))
        else:
            coords = np.column_stack((x_coords, y_coords))

        logger.info(
            "Generating tetrahedral connectivity at"
            f" {temporal_id}/{time}. This may take a few minutes and"
            " may even fail depending on the grid structure."
        )
        delaunay = spatial.Delaunay(coords)
        data_at_id.create_dataset(
            "tetrahedral_connectivity",
            data=delaunay.simplices.flatten(),
        )

    h5file.close()


@click.command(
    name="generate-tetrahedral-connectivity",
    help=generate_tetrahedral_connectivity.__doc__,
)
@click.argument(
    "h5file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    nargs=1,
    required=True,
)
@click.option(
    "--subfile-name",
    "-d",
    help=(
        "Name of the volume data subfile in the H5 files. A '.vol' extension is"
        " added if needed. If unspecified, and the first H5 file contains only"
        " a single '.vol' subfile, choose that. Otherwise, list all '.vol'"
        " subfiles and exit."
    ),
)
@click.option(
    "--stride", default=1, type=int, help="View only every stride'th time step"
)
@click.option(
    "--start-time",
    type=float,
    help=(
        "The earliest time at which to start visualizing. The start-time "
        "value is included."
    ),
)
@click.option(
    "--stop-time",
    type=float,
    help=(
        "The time at which to stop visualizing. The stop-time value is "
        "included."
    ),
)
@click.option(
    "--coordinates",
    default="InertialCoordinates",
    show_default=True,
    help="The coordinates to use for visualization",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing tetrahedral connectivity.",
)
def generate_tetrahedral_connectivity_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    generate_tetrahedral_connectivity(**kwargs)


if __name__ == "__main__":
    configure_logging(log_level=logging.INFO)
    generate_tetrahedral_connectivity_command(
        help_option_names=["-h", "--help"]
    )
