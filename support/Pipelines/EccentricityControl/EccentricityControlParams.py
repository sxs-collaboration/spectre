#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import functools
import glob
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import click
import h5py
import numpy as np
import pandas as pd
import yaml

from SimulationSupport.EccentricityControl.EccentricityControlParams import (
    EccentricityParams,
)
from SimulationSupport.EccentricityControl.EccentricityControlParams import (
    eccentricity_control_params as compute_eccentricity_control_params,
)
from spectre.IO.H5 import to_dataframe
from spectre.support.Yaml import SafeDumper
from spectre.Visualization.PlotTrajectories import import_A_and_B

logger = logging.getLogger(__name__)

DEFAULT_AHA_TRAJECTORIES = "ApparentHorizons/ControlSystemAhA_Centers.dat"
DEFAULT_AHB_TRAJECTORIES = "ApparentHorizons/ControlSystemAhB_Centers.dat"


def eccentricity_control_params(
    h5_files: Union[Union[str, Path], Sequence[Union[str, Path]]],
    id_input_file_path: Union[str, Path],
    subfile_name_aha_trajectories: str = DEFAULT_AHA_TRAJECTORIES,
    subfile_name_ahb_trajectories: str = DEFAULT_AHB_TRAJECTORIES,
    subfile_name_aha_quantities: str = "ObservationAhA.dat",
    subfile_name_ahb_quantities: str = "ObservationAhB.dat",
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    plot_output_dir: Optional[Union[str, Path]] = None,
    ecc_params_output_file: Optional[Union[str, Path]] = None,
) -> Dict[EccentricityParams, float]:
    """Get new orbital parameters for a binary system to control eccentricity.

    The eccentricity is estimated from the trajectories of the binary objects
    and updates to the orbital parameters are suggested to drive the orbit to
    the target eccentricity, using the routines in the SimulationSupport
    package. Currently supports only circular target orbits (target
    eccentricity = 0).

    Arguments:
      h5_files: Paths to the H5 files containing the trajectory data (e.g.
        BbhReductions.h5). Can also be a glob pattern.
      id_input_file_path: Path to the initial data input file from which the
        evolution started. This file contains the initial data parameters that
        are being controlled.
      subfile_name_aha_trajectories: (Optional) Name of the subfile containing
        the apparent horizon centers for object A.
      subfile_name_ahb_trajectories: (Optional) Name of the subfile containing
        the apparent horizon centers for object B.
      subfile_name_aha_quantities: (Optional) Name of the subfile containing the
        quantities measured on apparent horizon A (masses and spins).
      subfile_name_ahb_quantities: (Optional) Name of the subfile containing the
        quantities measured on apparent horizon B (masses and spins).
      tmin: (Optional) The lower time bound for the eccentricity estimate.
        Used to remove initial junk and transients in the data. If unspecified,
        SimulationSupport will estimate it.
      tmax: (Optional) The upper time bound for the eccentricity estimate.
        A reasonable value would include 2-3 orbits.
        Default is '500 + 5 * pi / Omega0'.
      plot_output_dir: (Optional) Output directory for plots.
      ecc_params_output_file: (Optional) Output file for the results.

    Returns:
        Dictionary with the keys listed in 'EccentricityParams'.
    """
    # Make sure h5_files is a sequence
    if isinstance(h5_files, str):
        h5_files = glob.glob(h5_files)
        h5_files.sort()
    if isinstance(h5_files, Path):
        h5_files = [h5_files]

    # Read initial data parameters from input file
    with open(id_input_file_path, "r") as open_input_file:
        id_metadata, id_input_file = yaml.safe_load_all(open_input_file)
    target_params = id_metadata["TargetParams"]
    target_eccentricity = target_params["Eccentricity"]
    id_binary = id_input_file["Background"]["Binary"]
    Omega0 = id_binary["AngularVelocity"]
    adot0 = id_binary["Expansion"]
    D0 = id_binary["XCoords"][1] - id_binary["XCoords"][0]

    # Load trajectory data
    traj_A, traj_B = import_A_and_B(
        h5_files, subfile_name_aha_trajectories, subfile_name_ahb_trajectories
    )

    # Load horizon parameters from evolution data at the reference time
    def get_horizons_data(reductions_file):
        with h5py.File(reductions_file, "r") as open_h5file:
            horizons_data = []
            for ab, subfile_name in zip(
                "AB", [subfile_name_aha_quantities, subfile_name_ahb_quantities]
            ):
                ah_subfile = open_h5file.get(subfile_name)
                if ah_subfile is not None:
                    horizons_data.append(
                        to_dataframe(ah_subfile)
                        .set_index("Time")
                        .add_prefix(f"Ah{ab} ")
                    )
            if not horizons_data:
                return pd.DataFrame()
            return pd.concat(horizons_data, axis=1)

    horizon_params = pd.concat(map(get_horizons_data, h5_files))
    if horizon_params.empty:
        logger.warning(
            "No horizon data found. "
            "Using initial data masses and ignoring spins."
        )
        mA = target_params["MassA"]
        mB = target_params["MassB"]
        sA = sB = None
    else:
        mA = horizon_params["AhA ChristodoulouMass"].iloc[0]
        mB = horizon_params["AhB ChristodoulouMass"].iloc[0]
        if "AhA DimensionfulSpinVector_x" in horizon_params.columns:
            sA = np.array(
                [horizon_params.index]
                + [
                    horizon_params[f"AhA DimensionfulSpinVector_{xyz}"]
                    for xyz in "xyz"
                ]
            ).T
            sB = np.array(
                [horizon_params.index]
                + [
                    horizon_params[f"AhB DimensionfulSpinVector_{xyz}"]
                    for xyz in "xyz"
                ]
            ).T
        else:
            logger.warning("No horizon spins found in data, ignoring spins.")
            sA = sB = None

    # Estimate the eccentricity and compute updates to the orbital parameters
    ecc_params = compute_eccentricity_control_params(
        trajectory_a=traj_A,
        trajectory_b=traj_B,
        separation=D0,
        orbital_angular_velocity=Omega0,
        radial_expansion_velocity=adot0,
        mass_a=mA,
        mass_b=mB,
        spin_a=sA,
        spin_b=sB,
        tmin=tmin,
        tmax=tmax,
        target_eccentricity=target_eccentricity,
        plot_output_dir=plot_output_dir,
    )
    if ecc_params_output_file:
        with open(ecc_params_output_file, "w") as open_file:
            yaml.dump(ecc_params, open_file, Dumper=SafeDumper)
    return ecc_params


def eccentricity_control_params_options(f):
    """CLI options for the 'eccentricity_control_params' function.

    These options can be used by CLI commands that call the
    'eccentricity_control_params' function.
    """

    @click.argument(
        "h5_files",
        nargs=-1,
        type=click.Path(
            exists=True, file_okay=True, dir_okay=False, readable=True
        ),
    )
    @click.option(
        "--subfile-name-aha-trajectories",
        default=DEFAULT_AHA_TRAJECTORIES,
        show_default=True,
        help=(
            "Name of subfile containing the apparent horizon centers for"
            " object A."
        ),
    )
    @click.option(
        "--subfile-name-ahb-trajectories",
        default=DEFAULT_AHB_TRAJECTORIES,
        show_default=True,
        help=(
            "Name of subfile containing the apparent horizon centers for"
            " object B."
        ),
    )
    @click.option(
        "--subfile-name-aha-quantities",
        default="ObservationAhA.dat",
        show_default=True,
        help=(
            "Name of subfile containing the quantities measured on apparent"
            " horizon A (masses and spins)."
        ),
    )
    @click.option(
        "--subfile-name-ahb-quantities",
        default="ObservationAhB.dat",
        show_default=True,
        help=(
            "Name of subfile containing the quantities measured on apparent"
            " horizon A (masses and spins)."
        ),
    )
    @click.option(
        "--id-input-file",
        "-i",
        "id_input_file_path",
        required=True,
        help="Input file with initial data parameters.",
    )
    @click.option(
        "--tmin",
        type=float,
        help=(
            "The lower time bound for the eccentricity estimate. Used to remove"
            " initial junk and transients in the data."
        ),
    )
    @click.option(
        "--tmax",
        type=float,
        help=(
            "The upper time bound for the eccentricity estimate. A reasonable"
            " value would include 2-3 orbits."
        ),
    )
    @click.option(
        "--plot-output-dir",
        type=click.Path(writable=True),
        help="Output directory for plots.",
    )
    @click.option(
        "--ecc-params-output-file",
        type=click.Path(writable=True),
        help="Output file for the new orbital parameters.",
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@click.command(
    name="eccentricity-control-params", help=eccentricity_control_params.__doc__
)
@eccentricity_control_params_options
def eccentricity_control_params_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    eccentricity_control_params(**kwargs)
