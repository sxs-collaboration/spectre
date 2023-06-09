# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
import shutil

import click
import yaml


class MissingExpectedOutputError(Exception):
    def __init__(self, missing_files):
        self.missing_files = missing_files

    def __str__(self):
        return "Expected output files are missing: {}".format(
            self.missing_files
        )


def clean_output(input_file, output_dir, force):
    """
    Deletes output files specified in the `input_file` from the `output_dir`,
    raising an error if the expected output files were not found.

    The `input_file` must list its expected output files in the metadata:

    \b
    ```yaml
    ExpectedOutput:
      - Reduction.h5
      - Volume0.h5
    ```
    """
    with open(input_file, "r") as open_input_file:
        metadata = next(yaml.safe_load_all(open_input_file))

    if "ExpectedOutput" not in metadata:
        logging.warning(
            f"Input file {input_file} does not list 'ExpectedOutput' files."
        )
        return

    # Validate the user input. We have to be careful that we don't iterate over
    # a string, which would yield each character in turn.
    expected_output = metadata["ExpectedOutput"]
    assert not isinstance(expected_output, str), (
        f"'ExpectedOutput' in file '{input_file}' should be a list of files, "
        "not a string."
    )

    missing_files = []
    for expected_output_file in expected_output:
        expected_output_file = os.path.join(output_dir, expected_output_file)
        logging.debug(f"Attempting to remove file {expected_output_file}...")
        if os.path.exists(expected_output_file):
            if os.path.isfile(expected_output_file):
                os.remove(expected_output_file)
            else:
                shutil.rmtree(expected_output_file)
            logging.info(f"Removed file {expected_output_file}.")
        elif not force:
            missing_files.append(expected_output_file)
            logging.error(
                f"Expected file {expected_output_file} was not found."
            )
    # Raise an error if expected files were not found
    if len(missing_files) > 0:
        raise MissingExpectedOutputError(missing_files)


@click.command(help=clean_output.__doc__)
@click.argument(
    "input_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    required=True,
    help="Output directory of the run to clean up",
)
@click.option("--force", "-f", is_flag=True, help="Suppress all errors")
def clean_output_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    clean_output(**kwargs)


if __name__ == "__main__":
    clean_output_command(help_option_names=["-h", "--help"])
