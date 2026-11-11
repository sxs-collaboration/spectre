# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import click
import pandas as pd
import yaml

from spectre.Pipelines.Bbh.InitialData import generate_id
from spectre.Pipelines.Bbh.Inspiral import (
    INSPIRAL_INPUT_FILE_TEMPLATE,
    start_inspiral,
)
from spectre.Pipelines.EccentricityControl.EccentricityControlParams import (
    eccentricity_control_params,
    eccentricity_control_params_options,
)
from spectre.support.Schedule import scheduler_options

logger = logging.getLogger(__name__)


def eccentricity_control(
    h5_files: Union[Union[str, Path], Sequence[Union[str, Path]]],
    id_input_file_path: Union[str, Path],
    inspiral_input_file_path: Union[str, Path],
    pipeline_dir: Union[str, Path],
    # Eccentricity control parameters
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    plot_output_dir: Optional[Union[str, Path]] = None,
    ecc_params_output_file: Optional[Union[str, Path]] = None,
    # Options for continuing the evolution
    evolve: bool = True,
    branch_levs_when_complete: Optional[Sequence[int]] = None,
    inspiral_run_dir: Optional[Union[str, Path]] = None,
    inspiral_input_file_template: Union[
        str, Path
    ] = INSPIRAL_INPUT_FILE_TEMPLATE,
    # Scheduler options
    **scheduler_kwargs,
):
    """Adjust orbital parameters for eccentricity control.

    This function can be called after the inspiral has run (see the 'Next'
    section of the Inspiral.yaml file).

    Eccentricity control is carried out as an iterative process in which
    initial data for a binary system is constructed, and then the evolution
    of the inspiral is carried out for few orbits; after that, the data are
    analyzed, computing the eccentricity and predicting a new estimate of the
    orbital angular velocity and the radial expansion velocity that would
    produce a binary with a lower eccentricity; then, a new initial data is
    generated using such parameters and the process is repeated until the
    eccentricity falls below a prescribed absolute tolerance.

    To optimize the use of computing resources, the whole procedure can be
    carried out as a two-stage process. In the first, the rough stage, the
    evolution of the inspiral is carried out at a lower resolution. When the
    eccentricity has reached values below an absolute tolerance, the fine
    stage starts: the resolution is increased and the procedure is repeated.
    When eccentricity falls below an absolute tolerance tolerance (which may
    differ from the one used for the rough stage) eccentricity control is
    completed and the evolution of the inspiral is continued.

    This function does the following:

    - Reads orbital parameters from the 'id_input_file_path'.

    - Get the new orbital parameters by calling the function
      'eccentricity_control_params' in
      'spectre.Pipelines.EccentricityControl.EccentricityControl'.
      See this function for default values and more details on the arguments.

    - If the process is at the fine stage and the eccentricity is below an
      absolute tolerance, continue the evolution.

    - Otherwise set the new step for eccentricity reduction, eventually
      promoting from rough to fine stage, and generating new initial data based
      on updated orbital parameters with the 'generate_id' function.


    Arguments:
      h5_files: files that contain the trajectory data
      id_input_file_path: path to the input file of the initial data run
      pipeline_dir: Directory of the simulation, in which the pipeline
        creates its runs.
      evolve: Evolve the initial data after generation to continue eccentricity
        control. You can disable this to generate only the new initial data if
        you want to manually start the next inspiral.
      branch_levs_when_complete: Optional list of levs to start when
        eccentricity control is complete. Each lev continues in the 'Lev'
        subdirectory of the current eccentricity-control iteration, so the lev
        that was just run continues its sequence of segments and others branch
        into a new directory. If no levs are specified, the simulation will just
        stop after eccentricity control. See `Inspiral.INSPIRAL_LEVS` for the
        definition of the levs.
      inspiral_input_file_path: Path to the input file for the inspiral run. It
        is used to obtain the lev used from the run to extract the eccentricity
        from, which determines whether eccentricity control is at the rough or
        fine stage. It is also required if `branch_levs_when_complete` is
        specified, as the simulation will continue from this data.
      inspiral_run_dir: Directory where the inspiral run was executed.
        Defaults to the directory of the `inspiral_input_file_path`.
      inspiral_input_file_template: Input file template to start the
        `branch_levs_when_complete`. Defaults to the
        `INSPIRAL_INPUT_FILE_TEMPLATE`.

    See the 'eccentricity_control_params' function for details on the other
    arguments, as well as the 'schedule' function for the scheduling options.
    """
    # Read and process the initial data input file
    with open(id_input_file_path, "r") as open_input_file:
        id_metadata, id_input_file = yaml.safe_load_all(open_input_file)
    target_params = id_metadata["TargetParams"]
    assert (
        target_params["Eccentricity"] is not None
        and target_params["EccentricityAbsoluteTolerance"] is not None
    ), (
        "For eccentricity control the target eccentricity and its tolerance"
        " must be set."
    )

    # Read and process the input file for the inspiral, and extract the lev used
    with open(inspiral_input_file_path, "r") as inspiral_input_file:
        inspiral_metadata, inspiral_input_params = yaml.safe_load_all(
            inspiral_input_file
        )
    evolution_lev = inspiral_metadata["Lev"]
    assert (
        evolution_lev == target_params["RoughEccCtrlLev"]
        or evolution_lev == target_params["EvolutionLev"]
    ), (
        "Cannot determine the stage of eccentricity control: the evolution lev"
        " of the previous run mathches neither the rought nor the fine stage"
        " evolution lev."
    )

    # Find the current eccentricity and determine new parameters to put into
    # generate-id
    ecc_params = eccentricity_control_params(
        h5_files,
        id_input_file_path,
        tmin=tmin,
        tmax=tmax,
        plot_output_dir=plot_output_dir,
        ecc_params_output_file=ecc_params_output_file,
    )

    # If we are on the fine stage of eccentricity reduction and the eccentricity
    # is below the threshold we continue evolution
    if (
        evolution_lev == target_params["EvolutionLev"]
        and abs(ecc_params["Eccentricity"] - target_params["Eccentricity"])
        <= target_params["EccentricityAbsoluteTolerance"]
    ):
        logger.info("Eccentricity control complete.")
        if branch_levs_when_complete:
            # Continue the inspiral at each lev. The lev that we just ran
            # continues in the same directory, others branch into a new one.
            for lev in branch_levs_when_complete:
                start_inspiral(
                    # Start from inspiral data
                    id_input_file_path=inspiral_input_file_path,
                    id_run_dir=inspiral_run_dir,
                    id_subfile_name="PostJunkVolumeData",
                    lev=lev,
                    inspiral_input_file_template=inspiral_input_file_template,
                    continue_with_ringdown=True,
                    pipeline_dir=pipeline_dir,
                    **scheduler_kwargs,
                )
        return

    # Otherwise if we are on the rough stage of eccentricity reduction and the
    # eccentricity is below the rough threshold we promote to the fine stage
    if (
        evolution_lev == target_params["RoughEccCtrlLev"]
        and abs(ecc_params["Eccentricity"] - target_params["Eccentricity"])
        <= target_params["RoughEccentricityAbsoluteTolerance"]
    ):
        logger.info(
            "Eccentricity control completed on rough stage. "
            "Promoting to fine stage"
        )
        evolution_lev = target_params["EvolutionLev"]

    # Generate new initial data based on updated orbital parameters
    id_params = id_metadata["Next"]["With"]
    binary_data = id_input_file["Background"]["Binary"]
    x_B, x_A = binary_data["XCoords"]
    separation = x_A - x_B
    x_offset = x_A - target_params["MassB"] * separation
    y_offset, z_offset = binary_data["CenterOfMassOffset"]
    binary_domain = id_input_file["DomainCreator"]["BinaryCompactObject"]
    generate_id(
        target_params,
        # New orbital parameters
        separation=separation,
        orbital_angular_velocity=ecc_params["NewOmega0"],
        radial_expansion_velocity=ecc_params["NewAdot0"],
        # Initial guesses for ID control
        conformal_mass_a=binary_data["ObjectRight"]["KerrSchild"]["Mass"],
        conformal_mass_b=binary_data["ObjectLeft"]["KerrSchild"]["Mass"],
        horizon_rotation_a=binary_domain["ObjectA"]["Interior"][
            "ExciseWithBoundaryCondition"
        ]["ApparentHorizon"]["Rotation"],
        horizon_rotation_b=binary_domain["ObjectB"]["Interior"][
            "ExciseWithBoundaryCondition"
        ]["ApparentHorizon"]["Rotation"],
        center_of_mass_offset=[x_offset, y_offset, z_offset],
        linear_velocity=binary_data["LinearVelocity"],
        # Scheduling options
        refinement_level=id_params["control_refinement_level"],
        polynomial_order=id_params["control_polynomial_order"],
        control=True,
        evolve=evolve,
        evolution_lev=evolution_lev,
        eccentricity_control=True,
        pipeline_dir=pipeline_dir,
        **scheduler_kwargs,
    )


@click.command(name="eccentricity-control", help=eccentricity_control.__doc__)
@eccentricity_control_params_options
@click.option(
    "--pipeline-dir",
    "-d",
    type=click.Path(
        writable=True,
        path_type=Path,
    ),
    help="Directory of the simulation, in which the pipeline creates its runs.",
)
@click.option(
    "--evolve/--no-evolve",
    default=True,
    show_default=True,
    help=(
        "Evolve the initial data after generation to continue eccentricity "
        "control. You can disable this to generate only the new initial data "
        "if you want to manually start the next inspiral."
    ),
)
@scheduler_options
def eccentricity_control_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    eccentricity_control(**kwargs)
