# Distributed under the MIT License.
# See LICENSE.txt for details.

import click
import logging
import os
import re


class MissingExpectedOutputError(Exception):
    def __init__(self, missing_files):
        self.missing_files = missing_files

    def __str__(self):
        return "Expected output files are missing: {}".format(
            self.missing_files)


def clean_output(input_file, output_dir, force):
    """
    Deletes output files specified in the `input_file` from the `output_dir`,
    raising an error if the expected output files were not found.

    The `input_file` must list its expected output files in a comment, with the
    list of files indented by two spaces:

    \b
    ```yaml
    # ExpectedOutput:
    #   Reduction.h5
    #   Volume0.h5
    ```
    """
    found_indentation = None
    missing_files = []
    for line in open(input_file, 'r'):
        # Iterate through the file until we find the `ExpectedOutput` comment
        if found_indentation is None:
            matched_indentation = re.match('#([ ]*)ExpectedOutput:', line)
            if matched_indentation is not None:
                found_indentation = matched_indentation.groups()[0] + '  '
        else:
            # Now collect the output files listed in the comment.
            # We look for lines that are indented by two spaces relative to the
            # preceding `ExpectedOutput` comment
            matched_output_file = re.match('#' + found_indentation + '(.+)',
                                           line)
            if matched_output_file is None:
                logging.debug("Reached end of expected output file list.")
                break
            else:
                expected_output_file = os.path.join(
                    output_dir,
                    matched_output_file.groups()[0])
                logging.debug("Attempting to remove file {}...".format(
                    expected_output_file))
                if os.path.exists(expected_output_file):
                    os.remove(expected_output_file)
                    logging.info(
                        "Removed file {}.".format(expected_output_file))
                elif not force:
                    missing_files.append(expected_output_file)
                    logging.error("Expected file {} was not found.".format(
                        expected_output_file))
    if found_indentation is None:
        logging.warning(
            "Input file {} does not list `ExpectedOutput` files.".format(
                input_file))
    # Raise an error if expected files were not found
    if len(missing_files) > 0:
        raise MissingExpectedOutputError(missing_files)


@click.command(help=clean_output.__doc__)
@click.argument('input_file',
                type=click.Path(exists=True,
                                file_okay=True,
                                dir_okay=False,
                                readable=True))
@click.option('--output-dir',
              '-o',
              type=click.Path(exists=True,
                              file_okay=False,
                              dir_okay=True,
                              readable=True),
              required=True,
              help="Output directory of the run to clean up")
@click.option('--force', '-f', is_flag=True, help="Suppress all errors")
def clean_output_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    clean_output(**kwargs)


if __name__ == "__main__":
    clean_output_command(help_option_names=["-h", "--help"])
