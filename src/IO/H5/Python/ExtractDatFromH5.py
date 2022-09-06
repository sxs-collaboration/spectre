#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

from spectre.IO import H5 as spectre_h5
import spectre.DataStructures
import numpy as np
import os
import shutil


def extract_dat_files(filename, **kwargs):
    h5file = spectre_h5.H5File(filename, "r")

    all_dat_files = h5file.all_dat_files()

    if kwargs["list"]:
        print_str = "\n ".join(all_dat_files)
        print("Dat files within '{}':\n {}".format(filename, print_str))
        return

    out_dir = kwargs["output_directory"]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        if kwargs["force"]:
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

        if not os.path.exists(dat_dir):
            os.makedirs(dat_dir)

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


def parse_args():
    """
    Parse the command line arguments
    """
    import argparse as ap

    parser = ap.ArgumentParser(
        description=
        "Extract all Dat files inside a SpECTRE HDF5 file. The resulting files "
        "will be put into the '--output-directory'. The directory structure "
        "will be identical to the group structure inside the HDF5 file.")
    parser.add_argument('filename', help="The HDF5 file.")
    parser.add_argument(
        '--force',
        '-f',
        action='store_true',
        required=False,
        help="If an output directory already exists, overwrite it.")
    parser.add_argument('--list',
                        '-l',
                        action='store_true',
                        required=False,
                        help="List all dat files in the HDF5 file.")
    parser.add_argument(
        '--output-directory',
        '-o',
        required=False,
        help="Name of directory that will hold all the extracted Dat files. "
        "Default is 'extracted_FileName' where FileName is the name of the "
        "HDF5 file (without the '.h5' suffix).")

    args = parser.parse_args()

    if args.output_directory is None:
        # Remove ".h5" suffix
        args.output_directory = "extracted_" + args.filename[:-3]

    return args


if __name__ == "__main__":
    extract_dat_files(**vars(parse_args()))
