# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os

import click
import rich

import spectre.IO.H5 as spectre_h5


@click.command(name="combine-h5")
@click.option(
    "--file-prefix",
    required=True,
    type=str,
    help="prefix of the files to be combined (omit number and file extension)",
)
@click.option(
    "--subfile-name",
    "-d",
    type=str,
    help="subfile name of the volume file in the H5 file (omit file extension)",
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
    help="combined output filename (omit file extension)",
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
def combine_h5_command(file_prefix, subfile_name, output, check_src):
    """Combines multiple HDF5 volume files

    This executable is used for combining a series of HDF5 volume files into one
    continuous dataset to be stored in a single HDF5 volume file."""

    # Print available subfile names and exit
    filename = file_prefix + "0.h5"
    if not subfile_name:
        if os.path.exists(filename):
            spectre_file = spectre_h5.H5File(filename, "r")
            import rich.columns

            rich.print(rich.columns.Columns(spectre_file.all_vol_files()))
            return

    if subfile_name[0] == "/":
        subfile_name = subfile_name[1:]

    spectre_h5.combine_h5(file_prefix, subfile_name, output, check_src)


if __name__ == "__main__":
    combine_h5_command(help_option_names=["-h", "--help"])
