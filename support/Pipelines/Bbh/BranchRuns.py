# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import click

logger = logging.getLogger(__name__)

from spectre.Pipelines.Bbh.Inspiral import (
    INSPIRAL_INPUT_FILE_TEMPLATE,
    start_inspiral,
)
from spectre.support.Schedule import scheduler_options

# Levs defined in terms of p-refinement
ALL_LEVS = [
    {
        "Label": str(lev_number),
        "refinement_level": 1,
        "polynomial order": 7 + lev_number,
    }
    for lev_number in range(-2, 11)
]

# Levs 0 to 2
DEFAULT_LEVS = [lev for lev in ALL_LEVS if 0 <= int(lev["Label"]) <= 2]


def branch_runs(
    id_input_file_path: Union[str, Path],
    pipeline_dir: Union[str, Path],
    requested_levs: Optional[Sequence[int]] = None,
    id_run_dir: Optional[Union[str, Path]] = None,
    inspiral_input_file_template: Union[
        str, Path
    ] = INSPIRAL_INPUT_FILE_TEMPLATE,
    id_horizons_path: Optional[Union[str, Path]] = None,
    continue_with_ringdown: bool = False,
    **scheduler_kwargs,
):
    """Branch simulations into different Levs or resolutions.

      This function does the following:
      - Create subdirectories for the the pipelines associated to the different
        Levs or resolutions.
      - Generate input files and submit scripts for such runs.

      Arguments:
    id_input_file_path: path to the input file of the initial data or evolution
      run
    pipeline_dir: directory where the pipeline outputs are stored.
    requested_levs: Levs (resolutions) to evolve.
    id_run_dir: Directory of the initial data run. Paths in the input file
      are relative to this directory.
    inspiral_input_file_template: Input file template where parameters are
      inserted.
    id_horizons_path: Directory where the horizons data is stored.
    continue_with_ringdown: Flag to continue evolution into ringdown.

    """
    logger.warning(
        "The BBH pipeline is still experimental. Please review the"
        " generated input files."
    )

    if requested_levs is None:
        levs_for_branching = DEFAULT_LEVS
    else:
        # Select levs for branching
        min_lev = min(requested_levs)
        max_lev = max(requested_levs)
        assert (
            int(ALL_LEVS[0]["Label"])
            <= min_lev
            <= max_lev
            <= int(ALL_LEVS[-1]["Label"])
        ), "Lev range is undefined. Edit ALL_LEVS to add the missing Levs."
        levs_for_branching = [
            lev
            for lev in ALL_LEVS
            if int(lev["Label"]) in range(min_lev, max_lev + 1)
        ]

    if id_run_dir is None:
        id_run_dir = Path(id_input_file_path).resolve().parent

    # Resolve directories
    assert pipeline_dir is not None, (
        "Specify a '--pipeline-dir' / '-d' to continue with the ringdown"
        " simulation automatically. Don't specify a '--run-dir' / '-o' or"
        " '--segments-dir' / '-O' because it will be created in the"
        " 'pipeline_dir' automatically."
    )
    pipeline_dir = Path(pipeline_dir).resolve()

    job_name = scheduler_kwargs.get("job_name")

    for lev in levs_for_branching:
        lev_label = "Lev" + lev["Label"]
        lev_refinement_level = lev["refinement_level"]
        lev_polynomial_order = lev["polynomial order"]

        lev_dir = Path(f"{pipeline_dir}/{lev_label}")

        # Create the run directory
        logger.info(f"Configure run directory '{lev_dir}'")
        lev_dir.mkdir(parents=True, exist_ok=True)

        # Indicate lev in job name
        scheduler_kwargs["job_name"] = job_name + lev_label

        assert (scheduler_kwargs.get("run_dir") is None) and (
            scheduler_kwargs.get("segments_dir") is None
        ), (
            "Specify a '--pipeline-dir' / '-d' to continue. Don't specify a"
            " '--run-dir' / '-o' or '--segments-dir' / '-O' because it will"
            " be created in the 'pipeline_dir' automatically."
        )

        start_inspiral(
            id_input_file_path,
            refinement_level=lev_refinement_level,
            polynomial_order=lev_polynomial_order,
            id_run_dir=id_run_dir,
            inspiral_input_file_template=inspiral_input_file_template,
            id_horizons_path=id_horizons_path,
            continue_with_ringdown=continue_with_ringdown,
            eccentricity_control=False,
            pipeline_dir=lev_dir,
            **scheduler_kwargs,
        )


@click.command(name="branch-runs", help=branch_runs.__doc__)
@click.argument(
    "id_input_file_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "--pipeline-dir",
    "-d",
    type=click.Path(
        writable=True,
        path_type=Path,
    ),
    help="Directory where steps in the pipeline are created.",
)
@click.option(
    "--requested-levs",
    type=click.IntRange(-2, 10),
    nargs=2,
    help=(
        "Minimum and maximum Levs used for branching. LevN corresponds to a"
        " polynomial order of P=N+7."
    ),
)
@click.option(
    "-i",
    "--id-run-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    help=(
        "Directory of the initial data run. Paths in the input file are"
        " relative to this directory."
    ),
    show_default="directory of the ID_INPUT_FILE_PATH",
)
@click.option(
    "--inspiral-input-file-template",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=INSPIRAL_INPUT_FILE_TEMPLATE,
    help="Input file template for the inspiral.",
    show_default=True,
)
@click.option(
    "--id-horizons-path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=None,
    show_default="Horizons.h5 inside 'id-run-dir'",
    help=(
        "H5 file that holds information of the horizons of the ID solve. If"
        " this file does not exist in your ID directory, run 'spectre bbh"
        " postprocess-id' in the ID directory to generate it. Note that this is"
        " not needed if you are starting from a SpEC ID_Params.perl file."
    ),
)
@click.option(
    "--continue-with-ringdown",
    is_flag=True,
    help=(
        "Continue with the ringdown simulation once a common horizon has"
        " formed."
    ),
)
@scheduler_options
def branch_runs_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    branch_runs(**kwargs)


if __name__ == "__main__":
    branch_runs_command(help_option_names=["-h", "--help"])
