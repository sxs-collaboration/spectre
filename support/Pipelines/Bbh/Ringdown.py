# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Optional, Union

import click
import yaml
from rich.pretty import pretty_repr

from spectre.support.Schedule import schedule, scheduler_options

logger = logging.getLogger(__name__)

RINGDOWN_INPUT_FILE_TEMPLATE = Path(__file__).parent / "Ringdown.yaml"


def ringdown_parameters(
    inspiral_input_file: dict,
    inspiral_run_dir: Union[str, Path],
    refinement_level: int,
    polynomial_order: int,
) -> dict:
    """Determine ringdown parameters from the inspiral.

    These parameters fill the 'RINGDOWN_INPUT_FILE_TEMPLATE'.

    Arguments:
      inspiral_input_file: Inspiral input file as a dictionary.
      id_run_dir: Directory of the inspiral run. Paths in the input file
        are relative to this directory.
      refinement_level: h-refinement level.
      polynomial_order: p-refinement level.
    """
    return {
        # Initial data files
        "IdFileGlob": str(
            Path(inspiral_run_dir).resolve()
            / (inspiral_input_file["Observers"]["VolumeFileName"] + "*.h5")
        ),
        # Resolution
        "L": refinement_level,
        "P": polynomial_order,
    }


def start_ringdown(
    inspiral_input_file_path: Union[str, Path],
    refinement_level: int,
    polynomial_order: int,
    inspiral_run_dir: Optional[Union[str, Path]] = None,
    ringdown_input_file_template: Union[
        str, Path
    ] = RINGDOWN_INPUT_FILE_TEMPLATE,
    **scheduler_kwargs,
):
    """Schedule a ringdown simulation from the inspiral.

    Point the INSPIRAL_INPUT_FILE_PATH to the input file of the last inspiral
    segment. Also specify 'inspiral_run_dir' if the simulation was run in a
    different directory than where the input file is. Parameters for the
    ringdown will be determined from the inspiral and inserted into the
    'ringdown_input_file_template'. The remaining options are forwarded to the
    'schedule' command. See 'schedule' docs for details.
    """
    logger.warning(
        "The BBH pipeline is still experimental. Please review the"
        " generated input files."
    )

    # Determine ringdown parameters from inspiral
    with open(inspiral_input_file_path, "r") as open_input_file:
        _, inspiral_input_file = yaml.safe_load_all(open_input_file)
    if inspiral_run_dir is None:
        inspiral_run_dir = Path(inspiral_input_file_path).resolve().parent
    ringdown_params = ringdown_parameters(
        inspiral_input_file,
        inspiral_run_dir,
        refinement_level=refinement_level,
        polynomial_order=polynomial_order,
    )
    logger.debug(f"Ringdown parameters: {pretty_repr(ringdown_params)}")

    # Schedule!
    return schedule(
        ringdown_input_file_template, **ringdown_params, **scheduler_kwargs
    )


@click.command(name="start-ringdown", help=start_ringdown.__doc__)
@click.argument(
    "inspiral_input_file_path",
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
    "--inspiral-run-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    help=(
        "Directory of the last inspiral segment. Paths in the input file are"
        " relative to this directory."
    ),
    show_default="directory of the INSPIRAL_INPUT_FILE_PATH",
)
@click.option(
    "--ringdown-input-file-template",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=RINGDOWN_INPUT_FILE_TEMPLATE,
    help="Input file template for the ringdown.",
    show_default=True,
)
@click.option(
    "--refinement-level",
    "-L",
    type=int,
    help="h-refinement level.",
    default=0,
    show_default=True,
)
@click.option(
    "--polynomial-order",
    "-P",
    type=int,
    help="p-refinement level.",
    default=5,
    show_default=True,
)
@scheduler_options
def start_ringdown_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    start_ringdown(**kwargs)


if __name__ == "__main__":
    start_ringdown_command(help_option_names=["-h", "--help"])
