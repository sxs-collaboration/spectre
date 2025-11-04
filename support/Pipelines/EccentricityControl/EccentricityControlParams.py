#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import functools
import glob
import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import click
import h5py
import numpy as np
import pandas as pd
import yaml

from spectre.IO.H5 import to_dataframe
from spectre.support.CheckSpecImport import check_spec_import
from spectre.support.Yaml import SafeDumper
from spectre.Visualization.PlotTrajectories import import_A_and_B

logger = logging.getLogger(__name__)

# Keys of the dictionary returned by eccentricity_control_params
EccentricityParams = Literal[
    "Eccentricity",
    "EccentricityError",
    "Omega0",
    "Adot0",
    "D0",
    "DeltaOmega0",
    "DeltaAdot0",
    "DeltaD0",
    "NewOmega0",
    "NewAdot0",
    "NewD0",
    "Tmin",
    "Tmax",
]

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
    the target eccentricity, using SpEC's OmegaDotEccRemoval.py. Currently
    supports only circular target orbits (target eccentricity = 0).

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
        uses SpEC's 'OmegaDotEccRemoval.FindTmin' to estimate it.
      tmax: (Optional) The upper time bound for the eccentricity estimate.
        A reasonable value would include 2-3 orbits.
        Default is '500 + 5 * pi / Omega0'.
      plot_output_dir: (Optional) Output directory for plots.
      ecc_params_output_file: (Optional) Output file for the results.

    Returns:
        Dictionary with the keys listed in 'EccentricityParams'.
    """
    # Import functions from SpEC until we have ported them over
    check_spec_import(
        contains_commit="ecfabf1ce78daeacbdd026625a02215c8e84af0e",
    )
    from OmegaDotEccRemoval import (
        ComputeOmegaAndDerivsFromFile,
        FindTmin,
        performAllFits,
    )

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
    assert (
        target_eccentricity == 0.0
    ), "Only circular orbits are currently supported for eccentricity control."
    id_binary = id_input_file["Background"]["Binary"]
    Omega0 = id_binary["AngularVelocity"]
    adot0 = id_binary["Expansion"]
    D0 = id_binary["XCoords"][1] - id_binary["XCoords"][0]

    # Load trajectory data
    traj_A, traj_B = import_A_and_B(
        h5_files, subfile_name_aha_trajectories, subfile_name_ahb_trajectories
    )
    t, Omega, dOmegadt, OmegaVec = ComputeOmegaAndDerivsFromFile(traj_A, traj_B)

    # Set time bounds if not provided
    if tmin is None:
        tmin = max(FindTmin(t, dOmegadt, 500), t[0])
    if tmax is None:
        tmax = min(500 + 5 * np.pi / Omega0, t[-1])
    logger.info(
        "Estimating eccentricity from trajectory data in time range"
        f" {tmin:.3f} to {tmax:.3f}."
    )

    # Load horizon parameters from evolution data at reference time (tmin)
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

    # Call into SpEC's OmegaDotEccRemoval.py
    # Despite the name, this SpEC function takes arrays of data
    eccentricity, delta_Omega0, delta_adot0, delta_D0, ecc_std_dev, _ = (
        performAllFits(
            XA=traj_A,
            XB=traj_B,
            t=t,
            Omega=Omega,
            dOmegadt=dOmegadt,
            OmegaVec=OmegaVec,
            mA=mA,
            mB=mB,
            sA=sA,
            sB=sB,
            IDparam_omega0=Omega0,
            IDparam_adot0=adot0,
            IDparam_D0=D0,
            tmin=tmin,
            tmax=tmax,
            tref=tmin,
            opt_freq_filter=True,
            opt_varpro=True,
            opt_type="bbh",
            opt_tmin=tmin,
            opt_improved_Omega0_update=True,
            check_periastron_advance=True,
            plot_output_dir=plot_output_dir,
            Source="",
        )
    )
    logger.info(
        f"Eccentricity estimate is {eccentricity:g} +/- {ecc_std_dev:e}."
        " Update orbital parameters as follows"
        f" for target eccentricity {target_eccentricity:g} (choose two):\n"
        f"Omega0 += {delta_Omega0:e} -> {Omega0 + delta_Omega0:.8g}\n"
        f"adot0 += {delta_adot0:e} -> {adot0 + delta_adot0:e}\n"
        f"D0 += {delta_D0:e} -> {D0 + delta_D0:.8g}"
    )
    # These keys must correspond to 'EccentricityParams'
    ecc_params = {
        "Eccentricity": eccentricity,
        "EccentricityError": ecc_std_dev,
        "Omega0": Omega0,
        "Adot0": adot0,
        "D0": D0,
        "DeltaOmega0": delta_Omega0,
        "DeltaAdot0": delta_adot0,
        "DeltaD0": delta_D0,
        "NewOmega0": Omega0 + delta_Omega0,
        "NewAdot0": adot0 + delta_adot0,
        "NewD0": D0 + delta_D0,
        "Tmin": tmin,
        "Tmax": tmax,
    }
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
