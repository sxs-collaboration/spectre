#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.IO import H5 as spectre_h5
import click
import spectre.DataStructures
import numpy as np
import os
import shutil


def extract_dat_files(filename, out_dir, list=False, force=False):
    """Extract dat files from an H5 file

    Extract all Dat files inside a SpECTRE HDF5 file. The resulting files will
    be put into the 'OUT_DIR'. The directory structure will be identical to the
    group structure inside the HDF5 file.
    """
    h5file = spectre_h5.H5File(filename, "r")

    all_dat_files = h5file.all_dat_files()

    if list:
        print_str = "\n ".join(all_dat_files)
        print("Dat files within '{}':\n {}".format(filename, print_str))
        return

    if out_dir is None:
        raise ValueError(
            "An output directory is required unless listing file content.")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        if force:
            shutil.rmtree(out_dir, ignore_errors=True)
            os.mkdir(out_dir)
        else:
            raise ValueError(
                "Could not make directory '{}'. Already exists.".format(
                    out_dir))

    for dat_path in all_dat_files:
        split_path = dat_path.split("/")
        dat_dir = out_dir + "/".join(split_path[:-1])
        dat_filename = out_dir + dat_path

        os.makedirs(dat_dir, exist_ok=True)

        dat_file = h5file.get_dat(dat_path[:-4])

        legend = dat_file.get_legend()
        header = "\n".join("[{}] ".format(i) + "{}"
                           for i in range(len(legend))).format(*legend)

        dat_data = np.array(dat_file.get_data())

        np.savetxt(dat_filename,
                   dat_data,
                   delimiter=' ',
                   fmt="% .15e",
                   header=header)

        h5file.close()

    print("Successfully extracted all Dat files into '{}'".format(out_dir))


@click.command(help=extract_dat_files.__doc__)
@click.argument("filename",
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=False,
                                readable=True))
@click.argument("out_dir",
                type=click.Path(file_okay=False, dir_okay=True, writable=True),
                required=False)
@click.option('--force',
              '-f',
              is_flag=True,
              help="If the output directory already exists, overwrite it.")
@click.option('--list',
              '-l',
              is_flag=True,
              help="List all dat files in the HDF5 file and exit.")
def extract_dat_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    extract_dat_files(**kwargs)


if __name__ == "__main__":
    extract_dat_command(help_option_names=["-h", "--help"])
