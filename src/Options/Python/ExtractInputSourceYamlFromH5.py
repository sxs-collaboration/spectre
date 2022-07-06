#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import h5py
import argparse
import sys
import os
import logging


def read_input_source_from_h5(h5_path):
    '''
    Returns a string containing the InputSource.yaml attribute from h5file
    '''
    h5_file = h5py.File(h5_path, 'r')
    result = h5_file.attrs['InputSource.yaml'][0]
    h5_file.close()
    return result


def parse_cmd_line():
    '''
    parse command-line arguments
    :return: dictionary of the command-line args, dashes are underscores
    '''

    parser = argparse.ArgumentParser(
        description='Extract InputSource.yaml from an H5 file')
    parser.add_argument(
        '--h5file',
        type=str,
        required=True,
        help='Path to H5 file containing InputSource.yaml attribute')
    parser.add_argument('--output',
                        type=str,
                        required=True,
                        help='Path to output YAML-formatted text file')

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_cmd_line()
    input_source = read_input_source_from_h5(args.h5file)
    with open(args.output, 'w') as output_file:
        output_file.write(input_source)
