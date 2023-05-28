#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import multiprocessing as mp
import os
import shutil

import click
import h5py
import numpy as np
import pandas as pd

from spectre.Visualization.ReadH5 import available_subfiles


def write_dat_data(dat_path, h5_filename, out_dir, precision):
    with h5py.File(h5_filename, "r") as h5file:
        dat_file = h5file.get(dat_path)
        legend = dat_file.attrs["Legend"]
        dat_data = np.array(dat_file)

    header = "\n".join(f"[{i}] " + "{}" for i in range(len(legend))).format(
        *legend
    )

    format_string = f"%.{precision}e"

    # Write to stdout
    if not out_dir:
        print(
            pd.DataFrame(dat_data).to_string(
                index=False,
                header=legend,
                float_format=format_string,
                justify="left",
            )
        )
        return

    dat_dir = os.path.join(out_dir, os.path.dirname(dat_path))
    os.makedirs(dat_dir, exist_ok=True)
    dat_filename = os.path.join(out_dir, dat_path)

    np.savetxt(
        dat_filename, dat_data, delimiter=" ", fmt=format_string, header=header
    )


def get_all_dat_files(filename):
    with h5py.File(filename, "r") as h5file:
        return available_subfiles(h5file, extension=".dat")


def extract_dat_files(
    filename,
    out_dir,
    num_cores,
    precision,
    list=False,
    force=False,
    subfiles=None,
):
    """Extract dat files from an H5 file

    Extract all Dat files inside a SpECTRE HDF5 file. The resulting files will
    be put into the 'OUT_DIR'. The directory structure will be identical to the
    group structure inside the HDF5 file.
    """
    if list:
        print_str = "\n ".join(get_all_dat_files(filename))
        print(f"Dat files within '{filename}':\n {print_str}")
        return

    if subfiles:
        all_dat_files = subfiles
    else:
        all_dat_files = get_all_dat_files(filename)

    if out_dir:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        else:
            if force:
                shutil.rmtree(out_dir, ignore_errors=True)
                os.mkdir(out_dir)
            else:
                raise ValueError(
                    f"Could not make directory '{out_dir}'. Already exists."
                )
    else:
        if subfiles and len(subfiles) > 1:
            raise ValueError(
                "If no output directory is specified, can only write "
                f"one subfile to stdout, not {len(subfiles)}"
            )

    num_dat_files = len(all_dat_files)

    # Only use multiprocessing if we are writing to a directory and using more
    # than one core. Otherwise avoid the overhead
    if out_dir and num_cores > 1:
        with mp.Pool(processes=num_cores) as pool:
            pool.starmap(
                write_dat_data,
                zip(
                    all_dat_files,
                    [filename] * num_dat_files,
                    [out_dir] * num_dat_files,
                    [precision] * num_dat_files,
                ),
            )
    else:
        for dat_filename in all_dat_files:
            write_dat_data(dat_filename, filename, out_dir, precision)

    if out_dir:
        print(f"Successfully extracted all Dat files into '{out_dir}'")


@click.command(help=extract_dat_files.__doc__)
@click.argument(
    "filename",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.argument(
    "out_dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True),
    required=False,
)
@click.option(
    "--num-cores",
    "-j",
    default=1,
    show_default=True,
    help="Number of cores to run on.",
)
@click.option(
    "--precision",
    "-p",
    default=16,
    show_default=True,
    help="Precision with which to save (or print) the data.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="If the output directory already exists, overwrite it.",
)
@click.option(
    "--list",
    "-l",
    is_flag=True,
    help="List all dat files in the HDF5 file and exit.",
)
@click.option(
    "--subfile",
    "-d",
    "subfiles",
    multiple=True,
    help="Full path of subfile to extract (including extension).",
)
def extract_dat_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    extract_dat_files(**kwargs)


if __name__ == "__main__":
    extract_dat_command(help_option_names=["-h", "--help"])
