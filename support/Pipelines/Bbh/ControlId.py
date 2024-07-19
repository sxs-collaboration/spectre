# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import yaml

import spectre.IO.H5 as spectre_h5
from spectre.Pipelines.Bbh.InitialData import generate_id
from spectre.Visualization.ReadH5 import to_dataframe

logger = logging.getLogger(__name__)

DEFAULT_RESIDUAL_TOLERANCE = 1.0e-6
DEFAULT_MAX_ITERATIONS = 30


def control_id(
    id_input_file_path: Union[str, Path],
    id_run_dir: Optional[Union[str, Path]] = None,
    residual_tolerance: float = DEFAULT_RESIDUAL_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    refinement_level: int = 1,
    polynomial_order: int = 6,
):
    """Control BBH physical parameters.

    This function is called after initial data has been generated and horizons
    have been found in 'PostprocessId.py'. It uses an iterative scheme to drive
    the black hole physical parameters (masses and spins) closer to the desired
    values.

    For each iteration, this function does the following:

    - Determine new guesses for ID input parameters.

    - Generate initial data using these guesses.

    - Find horizons in the generated initial data.

    - Measure the difference between the horizon quantities and the desired
      values.

    Arguments:
      id_input_file_path: Path to the input file of the first initial data run.
      id_run_dir: Directory of the first initial data run. If not provided, the
        directory of the input file is used.
      residual_tolerance: Residual tolerance used for termination condition.
        (Default: 1.e-6)
      max_iterations: Maximum of iterations allowed. Note: each iteration is
        very expensive as it needs to solve an entire initial data problem.
        (Default: 30)
      refinement_level: h-refinement used in control loop.
      polynomial_order: p-refinement used in control loop.
    """

    # Read input file
    if id_run_dir is None:
        id_run_dir = Path(id_input_file_path).resolve().parent
    with open(id_input_file_path, "r") as open_input_file:
        _, id_input_file = yaml.safe_load_all(open_input_file)
    binary_data = id_input_file["Background"]["Binary"]
    M_target_A = binary_data["ObjectRight"]["KerrSchild"]["Mass"]
    M_target_B = binary_data["ObjectLeft"]["KerrSchild"]["Mass"]
    chi_target_A = binary_data["ObjectRight"]["KerrSchild"]["Spin"]
    chi_target_B = binary_data["ObjectLeft"]["KerrSchild"]["Spin"]
    x_B, x_A = binary_data["XCoords"]
    separation = x_A - x_B
    orbital_angular_velocity = binary_data["AngularVelocity"]
    radial_expansion_velocity = binary_data["Expansion"]

    # File to write control diagnostic data
    data_file = open(f"{id_run_dir}/ControlParamsData.txt", "w")

    iteration = 0
    control_run_dir = id_run_dir

    # Function to be minimized
    def Residual(
        M_Kerr_A: float,
        M_Kerr_B: float,
        chi_Kerr_A_x: float,
        chi_Kerr_A_y: float,
        chi_Kerr_A_z: float,
        chi_Kerr_B_x: float,
        chi_Kerr_B_y: float,
        chi_Kerr_B_z: float,
    ):
        nonlocal iteration
        nonlocal control_run_dir

        if iteration > 0:
            logger.info(
                "\n"
                "=========================================="
                f" Control of BBH Parameters ({iteration}) "
                "=========================================="
            )
            control_run_dir = f"{id_run_dir}/ControlParams{iteration:02}"

            # Run ID and find horizons
            generate_id(
                mass_a=M_Kerr_A,
                mass_b=M_Kerr_B,
                dimensionless_spin_a=[chi_Kerr_A_x, chi_Kerr_A_y, chi_Kerr_A_z],
                dimensionless_spin_b=[chi_Kerr_B_x, chi_Kerr_B_y, chi_Kerr_B_z],
                separation=separation,
                orbital_angular_velocity=orbital_angular_velocity,
                radial_expansion_velocity=radial_expansion_velocity,
                run_dir=control_run_dir,
                control=False,
                evolve=False,
                scheduler=None,
                refinement_level=refinement_level,
                polynomial_order=polynomial_order,
            )

        # Get black hole physical parameters
        with spectre_h5.H5File(
            f"{control_run_dir}/Horizons.h5", "r"
        ) as horizons_file:
            AhA_quantities = to_dataframe(
                horizons_file.get_dat("AhA.dat")
            ).iloc[-1]

            M_A = AhA_quantities["ChristodoulouMass"]
            chi_A_x = AhA_quantities["DimensionlessSpinVector_x"]
            chi_A_y = AhA_quantities["DimensionlessSpinVector_y"]
            chi_A_z = AhA_quantities["DimensionlessSpinVector_z"]

            horizons_file.close_current_object()
            AhB_quantities = to_dataframe(
                horizons_file.get_dat("AhB.dat")
            ).iloc[-1]

            M_B = AhB_quantities["ChristodoulouMass"]
            chi_B_x = AhB_quantities["DimensionlessSpinVector_x"]
            chi_B_y = AhB_quantities["DimensionlessSpinVector_y"]
            chi_B_z = AhB_quantities["DimensionlessSpinVector_z"]

        residual = np.array(
            [
                M_A - M_target_A,
                M_B - M_target_B,
                chi_A_x - chi_target_A[0],
                chi_A_y - chi_target_A[1],
                chi_A_z - chi_target_A[2],
                chi_B_x - chi_target_B[0],
                chi_B_y - chi_target_B[1],
                chi_B_z - chi_target_B[2],
            ]
        )
        logger.info(f"Control Residual = {np.max(np.abs(residual)):e}")

        data_file.write(
            f" {iteration},"
            f" {M_A}, {M_B},"
            f" {chi_A_x}, {chi_A_y}, {chi_A_z},"
            f" {chi_B_x}, {chi_B_y}, {chi_B_z},"
            f" {residual[0]}, {residual[1]},"
            f" {residual[2]}, {residual[3]}, {residual[4]},"
            f" {residual[5]}, {residual[6]}, {residual[7]}"
            " \n"
        )
        data_file.flush()

        return residual

    # Initial guess
    u = np.array([M_target_A, M_target_B, *chi_target_A, *chi_target_B])

    # Initial residual
    F = Residual(*u)

    # Initial Jacobian (identity matrix)
    J = np.identity(len(u))

    while iteration < max_iterations and np.max(np.abs(F)) > residual_tolerance:
        iteration += 1

        # Update the free parameters using a quasi-Newton-Raphson method
        Delta_u = -np.dot(np.linalg.inv(J), F)
        u += Delta_u

        F = Residual(*u)

        # Update the Jacobian using Broyden's method
        J += np.outer(F, Delta_u) / np.dot(Delta_u, Delta_u)

    data_file.close()

    return control_run_dir
