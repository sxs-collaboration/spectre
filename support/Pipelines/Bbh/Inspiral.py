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

INSPIRAL_INPUT_FILE_TEMPLATE = Path(__file__).parent / "Inspiral.yaml"


def inspiral_parameters(
    id_input_file: dict,
    id_run_dir: Union[str, Path],
    refinement_level: int,
    polynomial_order: int,
) -> dict:
    """Determine inspiral parameters from initial data.

    These parameters fill the 'INSPIRAL_INPUT_FILE_TEMPLATE'.

    Arguments:
      id_input_file: Initial data input file as a dictionary.
      id_run_dir: Directory of the initial data run. Paths in the input file
        are relative to this directory.
      refinement_level: h-refinement level.
      polynomial_order: p-refinement level.
    """
    id_domain_creator = id_input_file["DomainCreator"]["BinaryCompactObject"]
    id_binary = id_input_file["Background"]["Binary"]
    return {
        # Initial data files
        "IdFileGlob": str(
            Path(id_run_dir).resolve()
            / (id_input_file["Observers"]["VolumeFileName"] + "*.h5")
        ),
        # Domain geometry
        "ExcisionRadiusA": id_domain_creator["ObjectA"]["InnerRadius"],
        "ExcisionRadiusB": id_domain_creator["ObjectB"]["InnerRadius"],
        "XCoordA": id_domain_creator["ObjectA"]["XCoord"],
        "XCoordB": id_domain_creator["ObjectB"]["XCoord"],
        # Initial functions of time
        "InitialAngularVelocity": id_binary["AngularVelocity"],
        "RadialExpansionVelocity": float(id_binary["Expansion"]),
        # Resolution
        "L": refinement_level,
        "P": polynomial_order,
    }


def start_inspiral(
    id_input_file_path: Union[str, Path],
    refinement_level: int,
    polynomial_order: int,
    id_run_dir: Optional[Union[str, Path]] = None,
    inspiral_input_file_template: Union[
        str, Path
    ] = INSPIRAL_INPUT_FILE_TEMPLATE,
    continue_with_ringdown: bool = False,
    pipeline_dir: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    segments_dir: Optional[Union[str, Path]] = None,
    **scheduler_kwargs,
):
    """Schedule an inspiral simulation from initial data.

    Point the ID_INPUT_FILE_PATH to the input file of your initial data run.
    Also specify 'id_run_dir' if the initial data was run in a different
    directory than where the input file is. Parameters for the inspiral will be
    determined from the initial data and inserted into the
    'inspiral_input_file_template'. The remaining options are forwarded to the
    'schedule' command. See 'schedule' docs for details.

    ## Resource allocation

    Runs on 4 nodes by default when scheduled on a cluster. Set 'num_nodes' to
    adjust.
    """
    logger.warning(
        "The BBH pipeline is still experimental. Please review the"
        " generated input files."
    )

    # Determine inspiral parameters from initial data
    with open(id_input_file_path, "r") as open_input_file:
        _, id_input_file = yaml.safe_load_all(open_input_file)
    if id_run_dir is None:
        id_run_dir = Path(id_input_file_path).resolve().parent
    inspiral_params = inspiral_parameters(
        id_input_file,
        id_run_dir,
        refinement_level=refinement_level,
        polynomial_order=polynomial_order,
    )
    logger.debug(f"Inspiral parameters: {pretty_repr(inspiral_params)}")

    # Resolve directories
    if pipeline_dir:
        pipeline_dir = Path(pipeline_dir).resolve()
    if continue_with_ringdown:
        assert pipeline_dir is not None, (
            "Specify a '--pipeline-dir' / '-d' to continue with the ringdown"
            " simulation automatically. Don't specify a '--run-dir' / '-o' or"
            " '--segments-dir' / '-O' because it will be created in the"
            " 'pipeline_dir' automatically."
        )
        assert run_dir is None and segments_dir is None, (
            "Specify the '--pipeline-dir' / '-d' rather than '--run-dir' / '-o'"
            " or '--segments-dir' / '-O' when continuing with the ringdown"
            " simulation. Directories for the evolution will be created in the"
            " 'pipeline_dir' automatically."
        )
    if pipeline_dir and not run_dir and not segments_dir:
        segments_dir = pipeline_dir / "002_Inspiral"

    # Determine resource allocation
    if (
        scheduler_kwargs.get("scheduler") is not None
        and scheduler_kwargs.get("num_procs") is None
        and scheduler_kwargs.get("num_nodes") is None
    ):
        # Just run on 4 nodes for now, because 1 is surely not enough. We can
        # make this smarter later (e.g. scale with the number of elements).
        scheduler_kwargs["num_nodes"] = 4

    # Schedule!
    return schedule(
        inspiral_input_file_template,
        **inspiral_params,
        **scheduler_kwargs,
        continue_with_ringdown=continue_with_ringdown,
        pipeline_dir=pipeline_dir,
        run_dir=run_dir,
        segments_dir=segments_dir,
    )


@click.command(name="start-inspiral", help=start_inspiral.__doc__)
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
    "--refinement-level",
    "-L",
    type=int,
    help="h-refinement level.",
    default=1,
    show_default=True,
)
@click.option(
    "--polynomial-order",
    "-P",
    type=int,
    help="p-refinement level.",
    default=9,
    show_default=True,
)
@click.option(
    "--continue-with-ringdown",
    is_flag=True,
    help=(
        "Continue with the ringdown simulation once a common horizon has"
        " formed."
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
@scheduler_options
def start_inspiral_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    start_inspiral(**kwargs)


if __name__ == "__main__":
    start_inspiral_command(help_option_names=["-h", "--help"])
