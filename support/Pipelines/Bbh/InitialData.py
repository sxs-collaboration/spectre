# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Optional, Union

import click
import rich_click
from rich.pretty import pretty_repr

from spectre.support.Schedule import schedule, scheduler_options

logger = logging.getLogger(__name__)

ID_INPUT_FILE_TEMPLATE = Path(__file__).parent / "InitialData.yaml"


def id_parameters(
    mass_ratio: float,
    separation: float,
    orbital_angular_velocity: float,
    radial_expansion_velocity: float,
    refinement_level: int,
    polynomial_order: int,
):
    """Determine initial data parameters from options.

    These parameters fill the 'ID_INPUT_FILE_TEMPLATE'.

    Arguments:
      mass_ratio: Defined as q = M_A / M_B >= 1.
      separation: Coordinate separation D of the black holes.
      orbital_angular_velocity: Omega_0.
      radial_expansion_velocity: adot_0.
      refinement_level: h-refinement level.
      polynomial_order: p-refinement level.
    """

    # Sanity checks
    assert mass_ratio >= 1.0, "Mass ratio is defined to be >= 1.0."

    # Determine initial data parameters from options
    M_A = mass_ratio / (1.0 + mass_ratio)
    M_B = 1.0 / (1.0 + mass_ratio)
    x_A = separation / (1.0 + mass_ratio)
    x_B = x_A - separation
    return {
        "MassRight": M_A,
        "MassLeft": M_B,
        "XRight": x_A,
        "XLeft": x_B,
        "ExcisionRadiusRight": 0.89 * 2.0 * M_A,
        "ExcisionRadiusLeft": 0.89 * 2.0 * M_B,
        "OrbitalAngularVelocity": orbital_angular_velocity,
        "RadialExpansionVelocity": radial_expansion_velocity,
        # Resolution
        "L": refinement_level,
        "P": polynomial_order,
    }


def generate_id(
    mass_ratio: float,
    separation: float,
    orbital_angular_velocity: float,
    radial_expansion_velocity: float,
    refinement_level: int,
    polynomial_order: int,
    id_input_file_template: Union[str, Path] = ID_INPUT_FILE_TEMPLATE,
    evolve: bool = False,
    pipeline_dir: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    segments_dir: Optional[Union[str, Path]] = None,
    **scheduler_kwargs,
):
    """Generate initial data for a BBH simulation.

    Parameters for the initial data will be inserted into the
    'id_input_file_template'. The remaining options are forwarded to the
    'schedule' command. See 'schedule' docs for details.
    """
    logger.warning(
        "The BBH pipeline is still experimental. Please review the"
        " generated input files."
    )

    # Resolve directories
    if pipeline_dir:
        pipeline_dir = Path(pipeline_dir).resolve()
    assert segments_dir is None, (
        "Initial data generation doesn't use segments at the moment. Specify"
        " '--run-dir' / '-o' or '--pipeline-dir' / '-d' instead."
    )
    if evolve:
        assert pipeline_dir is not None, (
            "Specify a '--pipeline-dir' / '-d' to evolve the initial data."
            " Don't specify a '--run-dir' / '-o' because it will be created in"
            " the 'pipeline_dir' automatically."
        )
        assert run_dir is None, (
            "Specify the '--pipeline-dir' / '-d' rather than '--run-dir' / '-o'"
            " when evolving the initial data. Directories for the initial data,"
            " evolution, etc will be created in the 'pipeline_dir'"
            " automatically."
        )
    if pipeline_dir and not run_dir:
        run_dir = pipeline_dir / "001_InitialData"

    # Determine initial data parameters from options
    id_params = id_parameters(
        mass_ratio=mass_ratio,
        separation=separation,
        orbital_angular_velocity=orbital_angular_velocity,
        radial_expansion_velocity=radial_expansion_velocity,
        refinement_level=refinement_level,
        polynomial_order=polynomial_order,
    )
    logger.debug(f"Initial data parameters: {pretty_repr(id_params)}")

    # Schedule!
    return schedule(
        id_input_file_template,
        **id_params,
        **scheduler_kwargs,
        evolve=evolve,
        pipeline_dir=pipeline_dir,
        run_dir=run_dir,
        segments_dir=segments_dir,
    )


@click.command(name="generate-id", help=generate_id.__doc__)
@click.option(
    "--mass-ratio",
    "-q",
    type=float,
    help="Mass ratio of the binary, defined as q = M_A / M_B >= 1.",
    required=True,
)
@click.option(
    "--separation",
    "-D",
    type=float,
    help="Coordinate separation D of the black holes.",
    required=True,
)
@click.option(
    "--orbital-angular-velocity",
    "-w",
    type=float,
    help="Orbital angular velocity Omega_0.",
    required=True,
)
@click.option(
    "--radial-expansion-velocity",
    "-a",
    type=float,
    help=(
        "Radial expansion velocity adot0 which is radial velocity over radius."
    ),
    required=True,
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
@click.option(
    "--id-input-file-template",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=ID_INPUT_FILE_TEMPLATE,
    help="Input file template for the initial data.",
    show_default=True,
)
@click.option(
    "--evolve",
    is_flag=True,
    help="Evolve the initial data after generation.",
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
def generate_id_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    generate_id(**kwargs)


rich_click.rich_click.OPTION_GROUPS["spectre bbh generate-id"] = [
    dict(
        name="Initial data parameters",
        options=[
            "--mass-ratio",
            "--separation",
            "--orbital-angular-velocity",
            "--radial-expansion-velocity",
        ],
    ),
    dict(
        name="Resolution", options=["--refinement-level", "--polynomial-order"]
    ),
    dict(
        name="Next",
        options=["--evolve"],
    ),
]


if __name__ == "__main__":
    generate_id_command(help_option_names=["-h", "--help"])
