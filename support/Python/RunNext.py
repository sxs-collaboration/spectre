# Distributed under the MIT License.
# See LICENSE.txt for details.

import contextlib
import importlib
import logging
import os
from pathlib import Path

import click
import yaml

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def working_directory(cwd: Path):
    """Temporarily change the working directory to 'cwd'."""
    prev_cwd = Path.cwd()
    os.chdir(cwd)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def run_next(next_entrypoint: dict, input_file_path: Path, cwd: Path):
    """Run the next entrypoint specified in the input file metadata

    Invokes the Python function specified in the 'Next' section of the input
    file metadata. It can be specified like this:

    \b
    ```yaml
    # Input file metadata
    Next:
      Run: spectre.Pipelines.Bbh.Inspiral:start_inspiral
      With:
        # Arguments to pass to the function
        submit: True
    ---
    # Rest of the input file
    ```

    The function will be invoked in the `cwd` directory ('--input-run-dir' /
    '-i'), which defaults to the directory of the input file. The following
    special values can be used for the arguments:

    - '__file__': The (absolute) path of the input file.
    - 'None': The Python value None.

    \f
    Arguments:
      next_entrypoint: The Python function to run. Must be a dictionary with
        the following keys:
        - "Run": The Python module and function to run, separated by a colon.
          For example, "spectre.Pipelines.Bbh.Ringdown:start_ringdown".
        - "With": A dictionary of arguments to pass to the function.
      input_file_path: Path to the input file that specified the entrypoint.
        Used to resolve '__file__' in the entrypoint arguments.
      cwd: The working directory in which to run the entrypoint. Used to
        resolve relative paths in the entrypoint arguments.
    """
    next_function = next_entrypoint["Run"].split(":")
    next_function = getattr(
        importlib.import_module(next_function[0]), next_function[1]
    )
    next_args = next_entrypoint["With"]
    substitutions = {
        "__file__": input_file_path.resolve(),
        "None": None,
    }
    next_args = {
        key: (
            substitutions.get(value, value) if isinstance(value, str) else value
        )
        for key, value in next_args.items()
    }
    with working_directory(cwd):
        return next_function(**next_args)


@click.command(name="run-next", help=run_next.__doc__)
@click.argument(
    "input_file_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "-i",
    "--input-run-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    help=(
        "Directory where the input file ran. Paths in the input file are"
        " relative to this directory."
    ),
    show_default="directory of the INPUT_FILE_PATH",
)
def run_next_command(input_file_path, input_run_dir):
    if input_run_dir is None:
        input_run_dir = Path(input_file_path).resolve().parent
    with open(input_file_path) as open_input_file:
        metadata = next(yaml.safe_load_all(open_input_file))
    if "Next" not in metadata:
        logger.info(
            "The input file metadata lists no 'Next' entrypoint. Nothing to do."
        )
        return
    run_next(
        metadata["Next"], input_file_path=input_file_path, cwd=input_run_dir
    )


if __name__ == "__main__":
    run_next_command(help_option_names=["-h", "--help"])
