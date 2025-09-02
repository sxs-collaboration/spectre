# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Union

import click
import numpy as np
import yaml
from rich.pretty import pretty_repr

from spectre.Pipelines.EccentricityControl.InitialOrbitalParameters import (
    initial_orbital_parameters,
)
from spectre.support.DirectoryStructure import PipelineStep, list_pipeline_steps
from spectre.support.Schedule import schedule, scheduler_options

logger = logging.getLogger(__name__)

ID_INPUT_FILE_TEMPLATE = Path(__file__).parent / "InitialData.yaml"

TargetParams = Literal[
    "MassRatio",
    "MassA",
    "MassB",
    "DimensionlessSpinA",
    "DimensionlessSpinB",
    "CenterOfMass",
    "AdmLinearMomentum",
    "AdmMass",
    "AdmAngularMomentumZ",
    "Eccentricity",
    "EccentricityAbsoluteTolerance",
    "MeanAnomalyFraction",
    "NumOrbits",
    "TimeToMerger",
]

DEFAULT_TARGET_PARAMS: Dict[TargetParams, float] = {
    "EccentricityAbsoluteTolerance": 1e-3,
}


def L1_distance(m1, m2, separation):
    """Distance of the L1 Lagrangian point from m1, in Newtonian gravity"""
    return separation * (0.5 - 0.227 * np.log10(m2 / m1))


def id_parameters(
    conformal_mass_a: float,
    conformal_mass_b: float,
    horizon_rotation_a: Sequence[float],
    horizon_rotation_b: Sequence[float],
    center_of_mass_offset: Sequence[float],
    linear_velocity: Sequence[float],
    separation: float,
    orbital_angular_velocity: float,
    radial_expansion_velocity: float,
    refinement_level: int,
    polynomial_order: int,
    negative_expansion_bc: bool,
    target_params: Dict[TargetParams, Union[float, Sequence[float]]],
):
    """Determine initial data parameters from options.

    These parameters fill the 'ID_INPUT_FILE_TEMPLATE'.

    Arguments:
      conformal_mass_a: Mass parameter of the larger black hole.
      conformal_mass_b: Mass parameter of the smaller black hole.
      horizon_rotation_a: Rotation parameter of the larger black hole, Omega_A.
      horizon_rotation_b: Rotation parameter of the smaller black hole, Omega_B.
      center_of_mass_offset: Offset from the Newtonian center of mass.
      linear_velocity: Velocity added to the shift boundary condition.
      separation: Coordinate separation D of the black holes.
      orbital_angular_velocity: Omega_0.
      radial_expansion_velocity: adot_0.
      refinement_level: h-refinement level.
      polynomial_order: p-refinement level.
      negative_expansion_bc: Place the excision boundaries inside of the
        apparent horizons.
      target_params: Target parameters for the initial data control loop.
    """
    for required_target in [
        "MassA",
        "MassB",
        "DimensionlessSpinA",
        "DimensionlessSpinB",
    ]:
        assert (
            required_target in target_params
        ), f"{required_target} must be specified in 'target_params'."

    x_A = (
        target_params["MassB"]
        / (target_params["MassA"] + target_params["MassB"])
        * separation
        + center_of_mass_offset[0]
    )
    x_B = x_A - separation

    # Spins
    chi_A = np.asarray(target_params["DimensionlessSpinA"])
    r_plus_A = conformal_mass_a * (1.0 + np.sqrt(1 - np.dot(chi_A, chi_A)))
    Omega_A = horizon_rotation_a
    Omega_A[2] += orbital_angular_velocity
    chi_B = np.asarray(target_params["DimensionlessSpinB"])
    r_plus_B = conformal_mass_b * (1.0 + np.sqrt(1 - np.dot(chi_B, chi_B)))
    Omega_B = horizon_rotation_b
    Omega_B[2] += orbital_angular_velocity
    if negative_expansion_bc:
        # For high spins, we need to place the excisions closer to the outer
        # horizon in order to avoid the inner horizon.
        excision_factor = (
            0.97
            if max(np.linalg.norm(chi_A), np.linalg.norm(chi_B)) > 0.9
            else 0.93
        )
    else:
        excision_factor = 1.0
    # Falloff widths of superposition
    L1_dist_A = L1_distance(conformal_mass_a, conformal_mass_b, separation)
    L1_dist_B = separation - L1_dist_A
    falloff_width_A = 3.0 / 5.0 * L1_dist_A
    falloff_width_B = 3.0 / 5.0 * L1_dist_B
    # This extra refinement was found through trial and error and allowed mass
    # ratio 6 to evolve through inspiral stably. This extra refinement doesn't
    # seem to scale linearly with mass ratio. The current hard-coded limits (3
    # and 5) were enough to find initial data for mass ratio 50 (no evolution
    # attempted).
    q = target_params["MassA"] / target_params["MassB"]
    extra_radial_refinement_l = min(round(q / 3.0) - 1 if (q > 3.0) else 0, 3)
    extra_radial_refinement_p = min(round(q / 5.0) if (q > 5.0) else 0, 5)
    horizon_l_max = (
        40 if max(np.linalg.norm(chi_A), np.linalg.norm(chi_B)) > 0.9 else 20
    )
    return {
        "ConformalMassRight": conformal_mass_a,
        "ConformalMassLeft": conformal_mass_b,
        "XRight": x_A,
        "XLeft": x_B,
        "CenterOfMassOffset_y": center_of_mass_offset[1],
        "CenterOfMassOffset_z": center_of_mass_offset[2],
        "LinearVelocity_x": linear_velocity[0],
        "LinearVelocity_y": linear_velocity[1],
        "LinearVelocity_z": linear_velocity[2],
        "ExcisionRadiusRight": excision_factor * r_plus_A,
        "ExcisionRadiusLeft": excision_factor * r_plus_B,
        "ObjectOuterRadius": separation / 3.75,
        "OrbitalAngularVelocity": orbital_angular_velocity,
        "RadialExpansionVelocity": radial_expansion_velocity,
        "ConformalSpinRight_x": chi_A[0],
        "ConformalSpinRight_y": chi_A[1],
        "ConformalSpinRight_z": chi_A[2],
        "ConformalSpinLeft_x": chi_B[0],
        "ConformalSpinLeft_y": chi_B[1],
        "ConformalSpinLeft_z": chi_B[2],
        "HorizonRotationRight_x": Omega_A[0],
        "HorizonRotationRight_y": Omega_A[1],
        "HorizonRotationRight_z": Omega_A[2],
        "HorizonRotationLeft_x": Omega_B[0],
        "HorizonRotationLeft_y": Omega_B[1],
        "HorizonRotationLeft_z": Omega_B[2],
        "FalloffWidthRight": falloff_width_A,
        "FalloffWidthLeft": falloff_width_B,
        "EnvelopeRadius": 4.0 * separation,
        # Resolution
        "L": refinement_level,
        "P": polynomial_order,
        "ExtraRadRef": extra_radial_refinement_l,
        "ExtraRadPoints": extra_radial_refinement_p,
        "HorizonLMax": horizon_l_max,
    }


def generate_id(
    target_params: Dict[TargetParams, Union[float, Sequence[float]]],
    # Orbital parameters
    separation: Optional[float] = None,
    orbital_angular_velocity: Optional[float] = None,
    radial_expansion_velocity: Optional[float] = None,
    # Control parameters
    conformal_mass_a: Optional[float] = None,
    conformal_mass_b: Optional[float] = None,
    horizon_rotation_a: Optional[Sequence[float]] = None,
    horizon_rotation_b: Optional[Sequence[float]] = None,
    center_of_mass_offset: Sequence[float] = [0.0, 0.0, 0.0],
    linear_velocity: Sequence[float] = [0.0, 0.0, 0.0],
    # Resolution
    refinement_level: int = 1,
    polynomial_order: int = 9,
    # Scheduling options
    id_input_file_template: Union[str, Path] = ID_INPUT_FILE_TEMPLATE,
    control: bool = False,
    evolve: bool = False,
    eccentricity_control: bool = False,
    negative_expansion_bc: bool = True,
    pipeline_dir: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
    segments_dir: Optional[Union[str, Path]] = None,
    out_file_name: str = "spectre.out",
    **scheduler_kwargs,
):
    """Generate initial data for a BBH simulation.

    Parameters for the initial data will be inserted into the
    'id_input_file_template'. The remaining options are forwarded to the
    'schedule' command. See 'schedule' docs for details.

    The 'target_params' are the parameters that the simulation should converge
    to. These values will never change during the simulation. Two control loops
    will try to drive the simulation towards these values, i.e., ID control
    (enable with 'control=True') and eccentricity control (enable with
    'eccentricity_control=True').

    ## ID control

    The ID control loop adjusts the conformal masses and spins to drive the
    horizon masses and spins to the specified values in a series of consecutive
    initial data solves. See 'support.Pipelines.Bbh.ControlId' for details. If
    unspecified, initial guesses for the conformal masses and spins default to
    the target masses and spins.

    ## Eccentricity control

    The eccentricity control loop adjusts the orbital parameters (initial
    coordinate separation D_0, orbital angular velocity Omega_0, and radial
    expansion velocity adot_0) to drive the eccentricity to the specified value
    in a series of short evolutions. See
    'support.Pipelines.Bbh.EccentricityControl' for details. If unspecified,
    initial guesses for the orbital parameters are obtained with the function
    'initial_orbital_parameters' in
    'support.Pipelines.EccentricityControl.InitialOrbitalParameters'.

    Scheduling options:
      id_input_file_template: Input file template where parameters are inserted.
      control: If set to True, a postprocessing control loop will adjust the
        input parameters to drive the horizon masses and spins to the specified
        values. If set to False, the horizon masses and spins in the generated
        data will differ from the input parameters. (default: False)
      evolve: Set to True to evolve the initial data after generation.
      eccentricity_control: If set to True, an eccentricity reduction script is
        run on the initial data to correct the initial orbital parameters.
      negative_expansion_bc: If set to True, the excision boundaries are set to
        be inside of the apparent horizons. This helps to find horizons and
        start an evolution from the initial data without extrapolation.
        (default: True)
      pipeline_dir: Directory where steps in the pipeline are created. Required
        when 'evolve' is set to True. The initial data will be created in a
        subdirectory '001_InitialData'.
      run_dir: Directory where the initial data is generated. Mutually exclusive
        with 'pipeline_dir'.
      segments_dir: Directory where the evolution data is generated. Mutually
        exclusive with 'pipeline_dir' and 'run_dir'.
      out_file_name: Optional. Name of the log file. (Default: "spectre.out")
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
    # If there is a pipeline directory, set run directory as well
    if pipeline_dir and not run_dir:
        pipeline_steps = list_pipeline_steps(pipeline_dir)
        if pipeline_steps:  # Check if the list is not empty
            run_dir = pipeline_steps[-1].next(label="InitialData").path
        else:
            run_dir = PipelineStep.first(
                directory=pipeline_dir, label="InitialData"
            ).path
    # If we run a control loop, then run initial data in a subdirectory
    if control:
        run_dir = f"{run_dir}/ControlParams_000"
        out_file_name = f"../{out_file_name}"

    # Determine orbital parameters
    if (
        separation is None
        or orbital_angular_velocity is None
        or radial_expansion_velocity is None
    ):
        separation, orbital_angular_velocity, radial_expansion_velocity = (
            initial_orbital_parameters(
                target_params,
                separation=separation,
                orbital_angular_velocity=orbital_angular_velocity,
                radial_expansion_velocity=radial_expansion_velocity,
            )
        )
    if eccentricity_control:
        assert (
            target_params["Eccentricity"] is not None
        ), "For eccentricity control the target eccentricity must be set."

    # This is an empirical factor based on an equal-mass non-spinning case, in
    # which ~0.41 conformal masses result in ~0.5 horizon masses.
    # mass_initial_guess_factor = 1.0
    mass_initial_guess_factor = 0.82
    if conformal_mass_a is None:
        conformal_mass_a = mass_initial_guess_factor * target_params["MassA"]
    if conformal_mass_b is None:
        conformal_mass_b = mass_initial_guess_factor * target_params["MassB"]

    # The 0.9 is an empirical factor that avoids an ill-posed elliptic problem
    # for very high spins.
    if horizon_rotation_a is None:
        chi_a = np.asarray(target_params["DimensionlessSpinA"])
        rotation_initial_guess_factor = (
            0.9 if np.linalg.norm(chi_a) > 0.99 else 1.0
        )
        horizon_rotation_a = (
            -rotation_initial_guess_factor
            * 0.5
            * chi_a
            / (conformal_mass_a * (1.0 + np.sqrt(1 - np.dot(chi_a, chi_a))))
        )
    if horizon_rotation_b is None:
        chi_b = np.asarray(target_params["DimensionlessSpinB"])
        rotation_initial_guess_factor = (
            0.9 if np.linalg.norm(chi_b) > 0.99 else 1.0
        )
        horizon_rotation_b = (
            -rotation_initial_guess_factor
            * 0.5
            * chi_b
            / (conformal_mass_b * (1.0 + np.sqrt(1 - np.dot(chi_b, chi_b))))
        )

    # Determine initial data parameters from options
    id_params = id_parameters(
        conformal_mass_a=conformal_mass_a,
        conformal_mass_b=conformal_mass_b,
        horizon_rotation_a=horizon_rotation_a,
        horizon_rotation_b=horizon_rotation_b,
        separation=separation,
        orbital_angular_velocity=orbital_angular_velocity,
        radial_expansion_velocity=radial_expansion_velocity,
        center_of_mass_offset=center_of_mass_offset,
        linear_velocity=linear_velocity,
        refinement_level=refinement_level,
        polynomial_order=polynomial_order,
        negative_expansion_bc=negative_expansion_bc,
        target_params=target_params,
    )
    logger.debug(f"Initial data parameters: {pretty_repr(id_params)}")

    # Store target parameters in the input file
    id_params["TargetParams"] = yaml.safe_dump(
        {"TargetParams": target_params}
    ).strip()

    # Schedule!
    return schedule(
        id_input_file_template,
        **id_params,
        **scheduler_kwargs,
        control=control,
        evolve=evolve,
        eccentricity_control=eccentricity_control,
        negative_expansion_bc=negative_expansion_bc,
        pipeline_dir=pipeline_dir,
        run_dir=run_dir,
        segments_dir=segments_dir,
        out_file_name=out_file_name,
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
    "--eccentricity-abs-tol",
    type=click.FloatRange(0.0, 1.0),
    default=DEFAULT_TARGET_PARAMS["EccentricityAbsoluteTolerance"],
    show_default=True,
    help="Absolute tolerance for eccentricity control.",
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
    default=9,
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
    "--eccentricity-control",
    is_flag=True,
    help=(
        "Perform eccentricity reduction script that finds current eccentricity"
        "and better guesses for the input orbital parameters."
    ),
)
@click.option(
    "--negative-expansion-bc/--no-negative-expansion-bc",
    default=True,
    show_default=True,
    help=(
        "Use negative expansion boundary condition so that the excision"
        "boundaries are inside of the apparent horizons. This helps to find"
        "horizons and start an evolution from the initial data without"
        "extrapolation."
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
    eccentricity_abs_tol,
    mean_anomaly_fraction,
    num_orbits,
    time_to_merger,
    **kwargs,
):
    _rich_traceback_guard = True  # Hide traceback until here
    target_params = {
        "MassRatio": mass_ratio,
        "MassA": mass_ratio / (1.0 + mass_ratio),
        "MassB": 1.0 / (1.0 + mass_ratio),
        "DimensionlessSpinA": dimensionless_spin_a,
        "DimensionlessSpinB": dimensionless_spin_b,
        "Eccentricity": eccentricity,
        "EccentricityAbsoluteTolerance": eccentricity_abs_tol,
        "MeanAnomalyFraction": mean_anomaly_fraction,
        "NumOrbits": num_orbits,
        "TimeToMerger": time_to_merger,
        "CenterOfMass": [0.0, 0.0, 0.0],
        "AdmLinearMomentum": [0.0, 0.0, 0.0],
    }
    if kwargs["eccentricity_control"]:
        # Only circular orbits are currently supported for eccentricity control,
        # so set target eccentricity to zero
        target_params["Eccentricity"] = 0.0

    generate_id(
        target_params,
        separation=separation,
        orbital_angular_velocity=orbital_angular_velocity,
        radial_expansion_velocity=radial_expansion_velocity,
        **kwargs,
    )


if __name__ == "__main__":
    generate_id_command(help_option_names=["-h", "--help"])
