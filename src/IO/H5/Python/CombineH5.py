# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os

import click
import rich

import spectre.IO.H5 as spectre_h5
from spectre.IO.H5.CombineH5Dat import combine_h5_dat_command


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
    "--check-src/--no-check-src",
    default=True,
    show_default=True,
    help=(
        "flag to check src files, True implies src files exist and can be"
        " checked, False implies no src files to check."
    ),
)
def combine_h5_vol_command(h5files, subfile_name, output, check_src):
    """Combines multiple HDF5 volume files

    This executable is used for combining a series of HDF5 volume files into one
    continuous dataset to be stored in a single HDF5 volume file."""
    # CLI scripts should be noops when input is empty
    if not h5files:
        return

    # Print available subfile names and exit
    if not subfile_name:
        spectre_file = spectre_h5.H5File(h5files[0], "r")
        import rich.columns

        rich.print(rich.columns.Columns(spectre_file.all_vol_files()))
        return

    if not subfile_name.startswith("/"):
        subfile_name = "/" + subfile_name
    if subfile_name.endswith(".vol"):
        subfile_name = subfile_name[:-4]

    if not output.endswith(".h5"):
        output += ".h5"

    spectre_h5.combine_h5(h5files, subfile_name, output, check_src)


if __name__ == "__main__":
    combine_h5_command(help_option_names=["-h", "--help"])
