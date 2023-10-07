# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Union

import click
from rich.pretty import pretty_repr

from spectre.support.Schedule import schedule, scheduler_options

logger = logging.getLogger(__name__)

ID_INPUT_FILE_TEMPLATE = Path(__file__).parent / "InitialData.yaml"


def id_parameters(
    mass_ratio: float,
    separation: float,
    orbital_angular_velocity: float,
    refinement_level: int,
    polynomial_order: int,
):
    """Determine initial data parameters from options.

    These parameters fill the 'ID_INPUT_FILE_TEMPLATE'.

    Arguments:
      mass_ratio: Defined as q = M_A / M_B >= 1.
      separation: Coordinate separation D of the black holes.
      orbital_angular_velocity: Omega_0.
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
        # Resolution
        "L": refinement_level,
        "P": polynomial_order,
    }


def generate_id(
    mass_ratio: float,
    separation: float,
    orbital_angular_velocity: float,
    refinement_level: int,
    polynomial_order: int,
    id_input_file_template: Union[str, Path] = ID_INPUT_FILE_TEMPLATE,
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

    # Determine initial data parameters from options
    id_params = id_parameters(
        mass_ratio=mass_ratio,
        separation=separation,
        orbital_angular_velocity=orbital_angular_velocity,
        refinement_level=refinement_level,
        polynomial_order=polynomial_order,
    )
    logger.debug(f"Initial data parameters: {pretty_repr(id_params)}")

    # Schedule!
    return schedule(id_input_file_template, **id_params, **scheduler_kwargs)


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
@scheduler_options
def generate_id_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    generate_id(**kwargs)


if __name__ == "__main__":
    generate_id_command(help_option_names=["-h", "--help"])
