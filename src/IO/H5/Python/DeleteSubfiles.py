# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
import shutil
import subprocess
import tempfile

import click
import h5py

logger = logging.getLogger(__name__)

HDF5_REPACK_EXECUTABLE = "@HDF5_REPACK_EXECUTABLE@"


@click.command()
@click.argument(
    "h5files",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    nargs=-1,
)
@click.option(
    "--subfile",
    "-d",
    "subfiles",
    required=True,
    multiple=True,
    help="Subfile to delete",
)
@click.option(
    "--repack/--no-repack",
    default=False,
    help=(
        "Repack the H5 files after deleting subfiles "
        "to reduce file size. Otherwise, the subfiles are deleted "
        "but the file size remains unchanged."
    ),
)
def delete_subfiles_command(h5files, subfiles, repack):
    """Delete subfiles from the 'H5FILES'"""
    for h5file in h5files:
        with h5py.File(h5file, "a") as open_h5_file:
            for subfile in subfiles:
                if subfile in open_h5_file:
                    del open_h5_file[subfile]
                    logger.debug(
                        f"Deleted subfile '{subfile}' from file: {h5file}"
                    )
                else:
                    logger.warning(f"No subfile '{subfile}' in file: {h5file}")
        if repack:
            # h5repack must write to a new file, so we create a temporary one
            with tempfile.TemporaryDirectory() as tempdir:
                tmp_h5file = os.path.join(tempdir, "temp.h5")
                subprocess.run(
                    [HDF5_REPACK_EXECUTABLE, h5file, tmp_h5file],
                    capture_output=True,
                    text=True,
                )
                shutil.move(tmp_h5file, h5file)
