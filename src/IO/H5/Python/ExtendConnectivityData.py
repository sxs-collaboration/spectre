# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging

import click
import rich

import spectre.IO.H5 as spectre_h5


@click.command(name="extend-connectivity")
@click.argument(
    "filename",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        writable=True,
    ),
)
@click.option(
    "--subfile-name",
    "-d",
    required=True,
    type=str,
    help="subfile name of the volume file in the H5 file (omit file extension)",
)
def extend_connectivity_data_command(filename, subfile_name):
    """Extend the connectivity inside a single HDF5 volume file.

    This extended connectivity is for some SpECTRE evolution, in
    order to fill in gaps between elements. Intended to be used as
    a post-processing routine to improve the quality of
    visualizations. Note: This does not work with subcell or AMR systems, and
    the connectivity only extends *within* each block and not between them.
    This only works for a single HDF5 volume file. If there are multiple
    files, the combine-h5 executable must be run first. The extend-connectivity
    command can then be run on the newly generated HDF5 file.
    """
    # Ensure that the format of the subfile is correct
    if subfile_name.endswith(".vol"):
        subfile_name = subfile_name[:-4]
    if subfile_name[0] != "/":
        subfile_name = "/" + subfile_name
    # Read the h5 file and extract the volume file from it
    h5_file = spectre_h5.H5File(filename, "r+")
    vol_file = h5_file.get_vol(subfile_name)

    observation_ids = vol_file.list_observation_ids()
    dim = vol_file.get_dimension()

    if dim == 1:
        vol_file.extend_connectivity_data_1d(observation_ids)
    elif dim == 2:
        vol_file.extend_connectivity_data_2d(observation_ids)
    elif dim == 3:
        vol_file.extend_connectivity_data_3d(observation_ids)
    else:
        raise ValueError("Invalid Dimensionality")


if __name__ == "__main__":
    extend_connectivity_data_command(help_option_names=["-h", "--help"])
