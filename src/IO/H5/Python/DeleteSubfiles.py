# Distributed under the MIT License.
# See LICENSE.txt for details.

import click
import h5py
import logging

logger = logging.getLogger(__name__)


@click.command()
@click.argument("h5files",
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=False,
                                readable=True),
                nargs=-1)
@click.option("--subfile",
              "-d",
              "subfiles",
              required=True,
              multiple=True,
              help="Subfile to delete")
def delete_subfiles_command(h5files, subfiles):
    """Delete subfiles from the 'H5FILES'"""
    for h5file in h5files:
        with h5py.File(h5file, "a") as open_h5_file:
            for subfile in subfiles:
                if subfile in open_h5_file:
                    del open_h5_file[subfile]
                    logger.debug(
                        f"Deleted subfile '{subfile}' from file: {h5file}")
                else:
                    logger.warning(f"No subfile '{subfile}' in file: {h5file}")
