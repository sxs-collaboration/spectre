# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os

import click
import rich

import spectre.IO.H5 as spectre_h5
from spectre.IO.H5.CombineH5Dat import combine_h5_dat_command
from spectre.support.CliExceptions import RequiredChoiceError


@click.group(name="combine-h5")
def combine_h5_command():
    """Combines multiple HDF5 files"""
    pass


combine_h5_command.add_command(combine_h5_dat_command, name="dat")


@combine_h5_command.command(name="vol")
@click.argument(
    "h5files",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    nargs=-1,
    required=True,
)
@click.option(
    "--subfile-name",
    "-d",
    help="subfile name of the volume file in the H5 file",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
    ),
    help="combined output filename",
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
        "not included."
    ),
)
@click.option(
    "--check-src/--no-check-src",
    default=True,
    show_default=True,
    help=(
        "flag to check src files, True implies src files exist and can be"
        " checked, False implies no src files to check."
    ),
)
def combine_h5_vol_command(
    h5files,
    subfile_name,
    output,
    start_time,
    stop_time,
    check_src,
):
    """Combines volume data spread over multiple H5 files into a single file

    The typical use case is to combine volume data from multiple nodes into a
    single file, if this is necessary for further processing (e.g. for the
    'extend-connectivity' command). Note that for most use cases it is not
    necessary to combine the volume data into a single file, as most commands
    can operate on multiple input H5 files (e.g. 'generate-xdmf').

    Note that this command does not currently combine volume data from different
    time steps (e.g. from multiple segments of a simulation). All input H5 files
    must contain the same set of observation IDs.
    """
    # Print available subfile names and exit
    if not subfile_name:
        spectre_file = spectre_h5.H5File(h5files[0], "r")
        raise RequiredChoiceError(
            (
                "Specify '--subfile-name' / '-d' to select a"
                " subfile containing volume data."
            ),
            choices=spectre_file.all_vol_files(),
        )

    if not output.endswith(".h5"):
        output += ".h5"

    spectre_h5.combine_h5(
        h5files,
        subfile_name,
        output,
        start_time,
        stop_time,
        check_src,
    )


if __name__ == "__main__":
    combine_h5_command(help_option_names=["-h", "--help"])
