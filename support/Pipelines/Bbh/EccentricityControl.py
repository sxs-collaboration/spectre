# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import click
import pandas as pd
import yaml

from spectre.Pipelines.Bbh.InitialData import generate_id
from spectre.Pipelines.EccentricityControl.EccentricityControlParams import (
    eccentricity_control_params,
    eccentricity_control_params_options,
)
from spectre.support.Schedule import scheduler_options

logger = logging.getLogger(__name__)


def eccentricity_control(
    h5_files: Union[Union[str, Path], Sequence[Union[str, Path]]],
    id_input_file_path: Union[str, Path],
    pipeline_dir: Union[str, Path],
    # Eccentricity control parameters
    tmin: Optional[float] = 500,
    tmax: Optional[float] = None,
    plot_output_dir: Optional[Union[str, Path]] = None,
    ecc_params_output_file: Optional[Union[str, Path]] = None,
    # Scheduler options
    evolve: bool = True,
    **scheduler_kwargs,
):
    """Eccentricity reduction post inspiral.

    This function can be called after the inspiral has run (see the 'Next'
    section of the Inspiral.yaml file).

    This function does the following:

    - Reads orbital parameters from the 'id_input_file_path'.

    - Sets the time boundaries for the eccentricity reduction process, starting
      at 500 and using all available data by default, with the option to adjust
      'tmin' and 'tmax' dynamically.

    - Get the new orbital parameters by calling the function
      'eccentricity_control_params' in
      'spectre.Pipelines.EccentricityControl.EccentricityControl'.

    - If the eccentricity is below a threshold, it prints "Success" and
      indicates that the simulation can continue.

    - Generates new initial data based on updated orbital parameters using the
      'generate_id' function.

    Arguments:
      h5_files: files that contain the trajectory data
      id_input_file_path: path to the input file of the initial data run
      pipeline_dir : directory where the pipeline outputs are stored.
      evolve: Evolve the initial data after generation to continue eccentricity
        control. You can disable this to generate only the new initial data if
        you want to manually start the next inspiral.

    See the 'eccentricity_control_params' function for details on the other
    arguments, as well as the 'schedule' function for the scheduling options.
    """
    # Read and process the initial data input file
    with open(id_input_file_path, "r") as open_input_file:
        id_metadata, id_input_file = yaml.safe_load_all(open_input_file)
    target_params = id_metadata["TargetParams"]
    assert (
        target_params["Eccentricity"] is not None
    ), "For eccentricity control the target eccentricity must be set."

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

    # Stop eccentricity control if eccentricity is below threshold
    if ecc_params["Eccentricity"] < 0.001:
        print("Success")
        # Should continue the simulation either by restarting from a
        # checkpoint, or from the volume data - will do later
        return

    # Generate new initial data based on updated orbital parameters
    id_params = id_metadata["Next"]["With"]
    binary_data = id_input_file["Background"]["Binary"]
    x_B, x_A = binary_data["XCoords"]
    separation = x_A - x_B
    x_offset = x_A - target_params["MassB"] * separation
    y_offset, z_offset = binary_data["CenterOfMassOffset"]
    generate_id(
        target_params,
        # New orbital parameters
        separation=separation,
        orbital_angular_velocity=ecc_params["NewOmega0"],
        radial_expansion_velocity=ecc_params["NewAdot0"],
        # Initial guesses for ID control
        conformal_mass_a=binary_data["ObjectRight"]["KerrSchild"]["Mass"],
        conformal_mass_b=binary_data["ObjectLeft"]["KerrSchild"]["Mass"],
        conformal_spin_a=binary_data["ObjectRight"]["KerrSchild"]["Spin"],
        conformal_spin_b=binary_data["ObjectLeft"]["KerrSchild"]["Spin"],
        center_of_mass_offset=[x_offset, y_offset, z_offset],
        linear_velocity=binary_data["LinearVelocity"],
        # Scheduling options
        refinement_level=id_params["control_refinement_level"],
        polynomial_order=id_params["control_polynomial_order"],
        control=True,
        evolve=evolve,
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
    help="Directory where steps in the pipeline are created.",
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
