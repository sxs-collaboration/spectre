# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Union

import h5py
import numpy as np
import yaml

import spectre.IO.H5 as spectre_h5
from spectre.Pipelines.Bbh.InitialData import generate_id
from spectre.Visualization.ReadH5 import to_dataframe

logger = logging.getLogger(__name__)

DEFAULT_RESIDUAL_TOLERANCE = 1.0e-6
DEFAULT_MAX_ITERATIONS = 30

# Initial data physical parameters that are supported in this control scheme
SupportedParams = Literal[
    "mass_A",
    "mass_B",
    "spin_A",
    "spin_B",
    "center_of_mass",
    "linear_momentum",
]

# Free data choices associated with each physical parameter
# Note 1: the values below need to match the argument names of `generate_id`.
# Note 2: mass_a/b and dimensionless_spin_a/b refer to the Kerr masses and spins
#         used in the background.
FreeDataFromParams: Dict[SupportedParams, str] = {
    "mass_A": "mass_a",
    "mass_B": "mass_b",
    "spin_A": "dimensionless_spin_a",
    "spin_B": "dimensionless_spin_b",
    "center_of_mass": "center_of_mass_offset",
    "linear_momentum": "linear_velocity",
}

# Quantites (free data or parameters) that are scalars
# Note: this is useful for switching between dictionaries and arrays below.
ScalarQuantities = ["mass_a", "mass_b", "mass_A", "mass_B"]


def control_id(
    id_input_file_path: Union[str, Path],
    control_params: Dict[SupportedParams, Union[float, Sequence[float]]],
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

    Suported control parameters (specified as keys of control_params):
      mass_A: Mass of the larger black hole.
      mass_B: Mass of the smaller black hole.
      spin_A: Dimensionless spin of the larger black hole.
      spin_B: Dimensionless spin of the smaller black hole.
      center_of_mass: Center of mass integral in general relativity.
      linear_momentum: ADM linear momentum.

    Example of control_params for an equal-mass non-spinning run with minimal
    drift of the center of mass:
      ```py
        control_params = dict(
          mass_A = 0.5,
          mass_B = 0.5,
          spin_A = [0., 0., 0.],
          spin_B = [0., 0., 0.],
          center_of_mass = [0., 0., 0.],
          linear_momentum = [0., 0., 0.],
        )
      ```

    Arguments:
      control_params: Dictionary used to customize control. The keys determine
        which physical parameters are controlled and the values determine their
        target result.
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

    assert (
        len(control_params) > 0
    ), "At least one control parameter must be specified."

    # Read input file
    if id_run_dir is None:
        id_run_dir = Path(id_input_file_path).resolve().parent
    with open(id_input_file_path, "r") as open_input_file:
        _, id_input_file = yaml.safe_load_all(open_input_file)
    binary_data = id_input_file["Background"]["Binary"]

    # Get initial xyz offset
    # Note: CenterOfMassOffset contains only the yz offsets, so we need to get
    # the x offset from XCoords
    mass_Kerr_A = binary_data["ObjectRight"]["KerrSchild"]["Mass"]
    mass_Kerr_B = binary_data["ObjectLeft"]["KerrSchild"]["Mass"]
    mass_ratio = mass_Kerr_A / mass_Kerr_B
    x_B, x_A = binary_data["XCoords"]
    separation = x_A - x_B
    x_offset = x_A - 1.0 / (1.0 + mass_ratio) * separation
    y_offset, z_offset = binary_data["CenterOfMassOffset"]

    # Combine initial choices of free data in a dictionary
    initial_free_data = dict(
        mass_a=mass_Kerr_A,
        mass_b=mass_Kerr_B,
        dimensionless_spin_a=binary_data["ObjectRight"]["KerrSchild"]["Spin"],
        dimensionless_spin_b=binary_data["ObjectLeft"]["KerrSchild"]["Spin"],
        center_of_mass_offset=[x_offset, y_offset, z_offset],
        linear_velocity=binary_data["LinearVelocity"],
    )

    # Get orbital velocities
    orbital_angular_velocity = binary_data["AngularVelocity"]
    radial_expansion_velocity = binary_data["Expansion"]

    # File to write control diagnostic data
    data_file = open(f"{id_run_dir}/ControlParamsData.txt", "w")

    iteration = 0
    control_run_dir = id_run_dir

    # Function to be minimized
    def Residual(u):
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

            # Start with initial free data choices and update the ones being
            # controlled in `control_params` with the numeric value from `u`
            free_data = initial_free_data.copy()
            u_iterator = iter(u)
            for key in [
                FreeDataFromParams[param] for param in control_params.keys()
            ]:
                if key in ScalarQuantities:
                    free_data[key] = next(u_iterator)
                else:
                    free_data[key] = [next(u_iterator) for _ in range(3)]

            # Run ID and find horizons
            generate_id(
                **free_data,
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

        # Initialize dictionary to hold the measured physical parameters
        measured_params: Dict[
            SupportedParams, Union[float, Sequence[float]]
        ] = {}

        # Get black hole physical parameters
        with spectre_h5.H5File(
            f"{control_run_dir}/Horizons.h5", "r"
        ) as horizons_file:
            AhA_quantities = to_dataframe(
                horizons_file.get_dat("AhA.dat")
            ).iloc[-1]

            if "mass_A" in control_params:
                measured_params["mass_A"] = AhA_quantities["ChristodoulouMass"]
            if "spin_A" in control_params:
                measured_params["spin_A"] = np.array(
                    [
                        AhA_quantities["DimensionlessSpinVector_x"],
                        AhA_quantities["DimensionlessSpinVector_y"],
                        AhA_quantities["DimensionlessSpinVector_z"],
                    ]
                )

            horizons_file.close_current_object()
            AhB_quantities = to_dataframe(
                horizons_file.get_dat("AhB.dat")
            ).iloc[-1]

            if "mass_B" in control_params:
                measured_params["mass_B"] = AhB_quantities["ChristodoulouMass"]
            if "spin_B" in control_params:
                measured_params["spin_B"] = np.array(
                    [
                        AhB_quantities["DimensionlessSpinVector_x"],
                        AhB_quantities["DimensionlessSpinVector_y"],
                        AhB_quantities["DimensionlessSpinVector_z"],
                    ]
                )

        # Get ADM integrals
        with spectre_h5.H5File(
            f"{control_run_dir}/BbhReductions.h5", "r"
        ) as reductions_file:
            adm_integrals = to_dataframe(
                reductions_file.get_dat("AdmIntegrals.dat")
            ).iloc[-1]

            if "center_of_mass" in control_params:
                measured_params["center_of_mass"] = np.array(
                    [
                        adm_integrals["CenterOfMass_x"],
                        adm_integrals["CenterOfMass_y"],
                        adm_integrals["CenterOfMass_z"],
                    ]
                )
            if "linear_momentum" in control_params:
                measured_params["linear_momentum"] = np.array(
                    [
                        adm_integrals["AdmLinearMomentum_x"],
                        adm_integrals["AdmLinearMomentum_y"],
                        adm_integrals["AdmLinearMomentum_z"],
                    ]
                )

        # Compute residual of physical parameters
        residual = np.array([])
        for key, target in control_params.items():
            if key in ScalarQuantities:
                residual = np.append(residual, [measured_params[key] - target])
            else:
                residual = np.append(residual, measured_params[key] - target)
        logger.info(f"Control Residual = {np.max(np.abs(residual)):e}")
        data_file.write(
            f" {iteration}, " + ", ".join(map(str, residual)) + " \n"
        )
        data_file.flush()

        return residual

    # Initial guess for free data
    u = np.array([])
    for key in [FreeDataFromParams[param] for param in control_params.keys()]:
        if key in ScalarQuantities:
            u = np.append(u, [initial_free_data[key]])
        else:
            u = np.append(u, initial_free_data[key])

    # Initial residual
    F = Residual(u)

    # Initialize Jacobian as an identity matrix
    J = np.identity(len(u))

    while iteration < max_iterations and np.max(np.abs(F)) > residual_tolerance:
        iteration += 1

        # Update the free parameters using a quasi-Newton-Raphson method
        Delta_u = -np.dot(np.linalg.inv(J), F)
        u += Delta_u

        F = Residual(u)

        # Update the Jacobian using Broyden's method
        J += np.outer(F, Delta_u) / np.dot(Delta_u, Delta_u)

    data_file.close()

    return control_run_dir
