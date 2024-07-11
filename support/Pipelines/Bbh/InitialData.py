# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import click
import numpy as np
from rich.pretty import pretty_repr

from spectre.Pipelines.EccentricityControl.InitialOrbitalParameters import (
    initial_orbital_parameters,
)
from spectre.support.Schedule import schedule, scheduler_options

logger = logging.getLogger(__name__)

ID_INPUT_FILE_TEMPLATE = Path(__file__).parent / "InitialData.yaml"


def L1_distance(m1, m2, separation):
    """Distance of the L1 Lagrangian point from m1, in Newtonian gravity"""
    return separation * (0.5 - 0.227 * np.log10(m2 / m1))


def id_parameters(
    mass_a: float,
    mass_b: float,
    dimensionless_spin_a: Sequence[float],
    dimensionless_spin_b: Sequence[float],
    separation: float,
    orbital_angular_velocity: float,
    radial_expansion_velocity: float,
    refinement_level: int,
    polynomial_order: int,
):
    """Determine initial data parameters from options.

    These parameters fill the 'ID_INPUT_FILE_TEMPLATE'.

    Arguments:
      mass_a: Mass of the larger black hole.
      mass_b: Mass of the smaller black hole.
      dimensionless_spin_a: Dimensionless spin of the larger black hole, chi_A.
      dimensionless_spin_b: Dimensionless spin of the smaller black hole, chi_B.
      separation: Coordinate separation D of the black holes.
      orbital_angular_velocity: Omega_0.
      radial_expansion_velocity: adot_0.
      refinement_level: h-refinement level.
      polynomial_order: p-refinement level.
    """

    mass_ratio = max(mass_a, mass_b) / min(mass_a, mass_b)

    # Determine initial data parameters from options
    M_A = mass_a
    M_B = mass_b
    x_A = separation / (1.0 + mass_ratio)
    x_B = x_A - separation
    # Spins
    chi_A = np.asarray(dimensionless_spin_a)
    r_plus_A = M_A * (1.0 + np.sqrt(1 - np.dot(chi_A, chi_A)))
    Omega_A = -0.5 * chi_A / r_plus_A
    Omega_A[2] += orbital_angular_velocity
    chi_B = np.asarray(dimensionless_spin_b)
    r_plus_B = M_B * (1.0 + np.sqrt(1 - np.dot(chi_B, chi_B)))
    Omega_B = -0.5 * chi_B / r_plus_B
    Omega_B[2] += orbital_angular_velocity
    # Falloff widths of superposition
    L1_dist_A = L1_distance(M_A, M_B, separation)
    L1_dist_B = separation - L1_dist_A
    falloff_width_A = 3.0 / 5.0 * L1_dist_A
    falloff_width_B = 3.0 / 5.0 * L1_dist_B
    return {
        "MassRight": mass_a,
        "MassLeft": mass_b,
        "XRight": x_A,
        "XLeft": x_B,
        "ExcisionRadiusRight": 0.93 * r_plus_A,
        "ExcisionRadiusLeft": 0.93 * r_plus_B,
        "OrbitalAngularVelocity": orbital_angular_velocity,
        "RadialExpansionVelocity": radial_expansion_velocity,
        "DimensionlessSpinRight_x": chi_A[0],
        "DimensionlessSpinRight_y": chi_A[1],
        "DimensionlessSpinRight_z": chi_A[2],
        "DimensionlessSpinLeft_x": chi_B[0],
        "DimensionlessSpinLeft_y": chi_B[1],
        "DimensionlessSpinLeft_z": chi_B[2],
        "HorizonRotationRight_x": Omega_A[0],
        "HorizonRotationRight_y": Omega_A[1],
        "HorizonRotationRight_z": Omega_A[2],
        "HorizonRotationLeft_x": Omega_B[0],
        "HorizonRotationLeft_y": Omega_B[1],
        "HorizonRotationLeft_z": Omega_B[2],
        "FalloffWidthRight": falloff_width_A,
        "FalloffWidthLeft": falloff_width_B,
        # Resolution
        "L": refinement_level,
        "P": polynomial_order,
    }


def generate_id(
    mass_a: float,
    mass_b: float,
    dimensionless_spin_a: Sequence[float],
    dimensionless_spin_b: Sequence[float],
    # Orbital parameters
    separation: float,
    orbital_angular_velocity: float,
    radial_expansion_velocity: float,
    # Resolution
    refinement_level: int = 1,
    polynomial_order: int = 6,
    # Scheduling options
    id_input_file_template: Union[str, Path] = ID_INPUT_FILE_TEMPLATE,
    control: bool = False,
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

    The orbital parameters can be computed with the function
    'initial_orbital_parameters' in
    'support.Pipelines.EccentricityControl.InitialOrbitalParameters'.

    Intrinsic parameters:
      mass_a: Mass of the larger black hole.
      mass_b: Mass of the smaller black hole.
      dimensionless_spin_a: Dimensionless spin of the larger black hole, chi_A.
      dimensionless_spin_b: Dimensionless spin of the smaller black hole, chi_B.

    Orbital parameters:
      separation: Coordinate separation D of the black holes.
      orbital_angular_velocity: Omega_0.
      radial_expansion_velocity: adot_0.

    Scheduling options:
      id_input_file_template: Input file template where parameters are inserted.
      control: If set to True, a postprocessing control loop will adjust the
        input parameters to drive the horizon masses and spins to the specified
        values. If set to False, the horizon masses and spins in the generated
        data will differ from the input parameters. (default: False)
      evolve: Set to True to evolve the initial data after generation.
      pipeline_dir: Directory where steps in the pipeline are created. Required
        when 'evolve' is set to True. The initial data will be created in a
        subdirectory '001_InitialData'.
      run_dir: Directory where the initial data is generated. Mutually exclusive
        with 'pipeline_dir'.
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
        mass_a=mass_a,
        mass_b=mass_b,
        dimensionless_spin_a=dimensionless_spin_a,
        dimensionless_spin_b=dimensionless_spin_b,
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
        control=control,
        evolve=evolve,
        pipeline_dir=pipeline_dir,
        run_dir=run_dir,
        segments_dir=segments_dir,
    )


@click.command(name="generate-id", help=generate_id.__doc__)
@click.option(
    "--mass-ratio",
    "-q",
    type=click.FloatRange(1.0, None),
    help="Mass ratio of the binary, defined as q = M_A / M_B >= 1.",
    required=True,
)
@click.option(
    "--dimensionless-spin-A",
    "--chi-A",
    type=click.FloatRange(-1.0, 1.0),
    nargs=3,
    help="Dimensionless spin of the larger black hole, chi_A.",
    required=True,
)
@click.option(
    "--dimensionless-spin-B",
    "--chi-B",
    type=click.FloatRange(-1.0, 1.0),
    nargs=3,
    help="Dimensionless spin of the smaller black hole, chi_B.",
    required=True,
)
# Orbital parameters
@click.option(
    "--separation",
    "-D",
    type=click.FloatRange(0.0, None, min_open=True),
    help="Coordinate separation D of the black holes.",
)
@click.option(
    "--orbital-angular-velocity",
    "-w",
    type=float,
    help="Orbital angular velocity Omega_0.",
)
@click.option(
    "--radial-expansion-velocity",
    "-a",
    type=float,
    help=(
        "Radial expansion velocity adot0 which is radial velocity over radius."
    ),
)
@click.option(
    "--eccentricity",
    "-e",
    type=click.FloatRange(0.0, 1.0),
    help=(
        "Eccentricity of the orbit. Specify together with _one_ of the other"
        " orbital parameters. Currently only an eccentricity of 0 is supported"
        " (circular orbit)."
    ),
)
@click.option(
    "--mean-anomaly-fraction",
    "-l",
    type=click.FloatRange(0.0, 1.0, max_open=True),
    help=(
        "Mean anomaly of the orbit divided by 2 pi, so it is a number between 0"
        " and 1. The value 0 corresponds to the pericenter of the orbit"
        " (closest approach), and the value 0.5 corresponds to the apocenter of"
        " the orbit (farthest distance)."
    ),
)
@click.option(
    "--num-orbits",
    type=click.FloatRange(0.0, None, min_open=True),
    help=(
        "Number of orbits until merger. Specify together with a zero"
        " eccentricity to compute initial orbital parameters for a circular"
        " orbit."
    ),
)
@click.option(
    "--time-to-merger",
    type=click.FloatRange(0.0, None, min_open=True),
    help=(
        "Time to merger. Specify together with a zero eccentricity to compute"
        " initial orbital parameters for a circular orbit."
    ),
)
# Resolution
@click.option(
    "--refinement-level",
    "-L",
    type=click.IntRange(0, None),
    help="h-refinement level.",
    default=1,
    show_default=True,
)
@click.option(
    "--polynomial-order",
    "-P",
    type=click.IntRange(1, None),
    help="p-refinement level.",
    default=6,
    show_default=True,
)
# Scheduling options
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
    "--control/--no-control",
    default=True,
    show_default=True,
    help="Control BBH physical parameters.",
)
@click.option(
    "--evolve",
    is_flag=True,
    help=(
        "Evolve the initial data after generation. When this flag"
        "is specified, you must also specify a pipeline directory (-d),"
        "instead of a run directory (-o)."
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
def generate_id_command(
    mass_ratio,
    dimensionless_spin_a,
    dimensionless_spin_b,
    separation,
    orbital_angular_velocity,
    radial_expansion_velocity,
    eccentricity,
    mean_anomaly_fraction,
    num_orbits,
    time_to_merger,
    **kwargs,
):
    _rich_traceback_guard = True  # Hide traceback until here
    # Determine orbital parameters
    separation, orbital_angular_velocity, radial_expansion_velocity = (
        initial_orbital_parameters(
            mass_ratio=mass_ratio,
            dimensionless_spin_a=dimensionless_spin_a,
            dimensionless_spin_b=dimensionless_spin_b,
            separation=separation,
            orbital_angular_velocity=orbital_angular_velocity,
            radial_expansion_velocity=radial_expansion_velocity,
            eccentricity=eccentricity,
            mean_anomaly_fraction=mean_anomaly_fraction,
            num_orbits=num_orbits,
            time_to_merger=time_to_merger,
        )
    )

    mass_a = mass_ratio / (1.0 + mass_ratio)
    mass_b = 1.0 / (1.0 + mass_ratio)

    generate_id(
        mass_a=mass_a,
        mass_b=mass_b,
        dimensionless_spin_a=dimensionless_spin_a,
        dimensionless_spin_b=dimensionless_spin_b,
        separation=separation,
        orbital_angular_velocity=orbital_angular_velocity,
        radial_expansion_velocity=radial_expansion_velocity,
        **kwargs,
    )


if __name__ == "__main__":
    generate_id_command(help_option_names=["-h", "--help"])
