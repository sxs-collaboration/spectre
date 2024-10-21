#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import functools
import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import click
import h5py
import numpy as np
import pandas as pd
import yaml

from spectre.Visualization.PlotTrajectories import import_A_and_B
from spectre.Visualization.ReadH5 import to_dataframe

logger = logging.getLogger(__name__)

# Input orbital parameters that can be controlled
OrbitalParams = Literal[
    "Omega0",
    "adot0",
    "D0",
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
    target_eccentricity: float = 0.0,
    target_mean_anomaly_fraction: Optional[float] = None,
    plot_output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[float, float, Dict[OrbitalParams, float]]:
    """Get new orbital parameters for a binary system to control eccentricity.

    The eccentricity is estimated from the trajectories of the binary objects
    and updates to the orbital parameters are suggested to drive the orbit to
    the target eccentricity, using SpEC's OmegaDotEccRemoval.py. Currently
    supports only circular target orbits (target eccentricity = 0).

    Arguments:
      h5_files: Paths to the H5 files containing the trajectory data (e.g.
        BbhReductions.h5).
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
        Used to remove initial junk and transients in the data.
      tmax: (Optional) The upper time bound for the eccentricity estimate.
        A reasonable value would include 2-3 orbits.
      target_eccentricity: (Optional) The target eccentricity to drive the
        orbit to. Default is 0.0 (circular orbit).
      target_mean_anomaly_fraction: (Optional) The target mean anomaly of the
        orbit divided by 2 pi, so it is a number between 0 and 1. The value 0
        corresponds to the pericenter of the orbit (closest approach), the value
        0.5 corresponds to the apocenter of the orbit (farthest distance), and
        the value 1 corresponds to the pericenter again. Currently this is
        unused because only an eccentricity of 0 is supported.
      plot_output_dir: (Optional) Output directory for plots.

    Returns:
        Tuple of eccentricity estimate, eccentricity error, and dictionary of
        new orbital parameters.
    """
    # Import functions from SpEC until we have ported them over
    try:
        from OmegaDotEccRemoval import (
            ComputeOmegaAndDerivsFromFile,
            performAllFits,
        )
    except ImportError:
        raise ImportError(
            "Importing from SpEC failed. Make sure you have pointed "
            "'-D SPEC_ROOT' to a SpEC installation when configuring the build "
            "with CMake."
        )

    # Make sure h5_files is a sequence
    if isinstance(h5_files, (str, Path)):
        h5_files = [h5_files]

    # Read initial data parameters from input file
    with open(id_input_file_path, "r") as open_input_file:
        _, id_input_file = yaml.safe_load_all(open_input_file)
    id_binary = id_input_file["Background"]["Binary"]
    Omega0 = id_binary["AngularVelocity"]
    adot0 = id_binary["Expansion"]
    D0 = id_binary["XCoords"][1] - id_binary["XCoords"][0]

    # Load trajectory data
    traj_A, traj_B = import_A_and_B(
        h5_files, subfile_name_aha_trajectories, subfile_name_ahb_trajectories
    )
    if tmin is not None:
        traj_A = traj_A[traj_A[:, 0] >= tmin]
        traj_B = traj_B[traj_B[:, 0] >= tmin]
    if tmax is not None:
        traj_A = traj_A[traj_A[:, 0] <= tmax]
        traj_B = traj_B[traj_B[:, 0] <= tmax]

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
    if tmin is not None:
        horizon_params = horizon_params[horizon_params.index >= tmin]
    if horizon_params.empty:
        logger.warning(
            "No horizon data found in time range. "
            "Using initial data masses and ignoring spins."
        )
        mA = id_binary["ObjectRight"]["KerrSchild"]["Mass"]
        mB = id_binary["ObjectLeft"]["KerrSchild"]["Mass"]
        sA = sB = None
    else:
        mA = horizon_params["AhA ChristodoulouMass"].iloc[0]
        mB = horizon_params["AhB ChristodoulouMass"].iloc[0]
        if "AhA DimensionfulSpinVector_x" in horizon_params.columns:
            sA = [
                horizon_params[f"AhA DimensionfulSpinVector_{xyz}"]
                for xyz in "xyz"
            ]
            sB = [
                horizon_params[f"AhB DimensionfulSpinVector_{xyz}"]
                for xyz in "xyz"
            ]
        else:
            logger.warning("No horizon spins found in data, ignoring spins.")
            sA = sB = None

    # Call into SpEC's OmegaDotEccRemoval.py
    t, Omega, dOmegadt, OmegaVec = ComputeOmegaAndDerivsFromFile(traj_A, traj_B)
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
            tmin=tmin or 0.0,
            tmax=tmax or np.inf,
            tref=tmin or 0.0,
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
    return (
        eccentricity,
        ecc_std_dev,
        {
            "Omega0": Omega0 + delta_Omega0,
            "adot0": adot0 + delta_adot0,
            "D0": D0 + delta_D0,
        },
    )


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
        default="ObservationsAhA.dat",
        show_default=True,
        help=(
            "Name of subfile containing the quantities measured on apparent"
            " horizon A (masses and spins)."
        ),
    )
    @click.option(
        "--subfile-name-ahb-quantities",
        default="ObservationsAhB.dat",
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
        "-o",
        "--plot-output-dir",
        type=click.Path(writable=True),
        help="Output directory for plots.",
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper
