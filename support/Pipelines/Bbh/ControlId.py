# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

import h5py
import numpy as np
import yaml

import spectre.IO.H5 as spectre_h5
from spectre.Pipelines.Bbh.InitialData import TargetParams, generate_id
from spectre.Visualization.ReadH5 import to_dataframe

logger = logging.getLogger(__name__)

# The default tolerance is just below the effect of junk radiation on the
# controlled paramaters, which  is about 1.0e-3.
DEFAULT_RESIDUAL_TOLERANCE = 1.0e-4
DEFAULT_MAX_ITERATIONS = 30

# Free data choices associated with each physical parameter
# Note 1: the values below need to match the argument names of `generate_id`.
# Note 2: conformal_mass_a/b and conformal_spin_a/b refer to the Kerr masses and
#         spins used in the background.
FreeDataFromParams: Dict[TargetParams, str] = {
    "MassA": "conformal_mass_a",
    "MassB": "conformal_mass_b",
    "DimensionlessSpinA": "conformal_spin_a",
    "DimensionlessSpinB": "conformal_spin_b",
    "CenterOfMass": "center_of_mass_offset",
    "AdmLinearMomentum": "linear_velocity",
    "AdmMass": "radial_expansion_velocity",
    "AdmAngularMomentumZ": "orbital_angular_velocity",
}

# Quantites (free data or parameters) that are scalars
# Note: this is useful for switching between dictionaries and arrays below.
ScalarQuantities = [
    "MassA",
    "MassB",
    "conformal_mass_a",
    "conformal_mass_b",
    "AdmMass",
    "AdmAngularMomentumZ",
    "radial_expansion_velocity",
    "orbital_angular_velocity",
]


def control_id(
    id_input_file_path: Union[str, Path],
    control_params: List[TargetParams],
    id_run_dir: Optional[Union[str, Path]] = None,
    residual_tolerance: float = DEFAULT_RESIDUAL_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    refinement_level: int = 1,
    polynomial_order: int = 6,
    negative_expansion_bc: bool = True,
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

    Supported control parameters:
      MassA: Mass of the larger black hole.
      MassB: Mass of the smaller black hole.
      DimensionlessSpinA: Dimensionless spin of the larger black hole.
      DimensionlessSpinB: Dimensionless spin of the smaller black hole.
      CenterOfMass: Center of mass integral in general relativity.
      AdmLinearMomentum: ADM linear momentum.

    A subset of these parameters can be chosen as the 'control_params'. The
    input file metadata must contain a 'TargetParams' dictionary with the
    corresponding target values.
    Example of control_params for an equal-mass non-spinning run with minimal
    drift of the center of mass:
    ```yaml
    TargetParams:
        MassA: 0.5
        MassB: 0.5
        DimensionlessSpinA: [0., 0., 0.]
        DimensionlessSpinB: [0., 0., 0.]
        CenterOfMass: [0., 0., 0.]
        AdmLinearMomentum: [0., 0., 0.]
    ```

    Arguments:
      control_params: List of parameters to control.
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
        id_metadata, id_input_file = yaml.safe_load_all(open_input_file)
    target_params = id_metadata["TargetParams"]
    binary_data = id_input_file["Background"]["Binary"]

    # Get initial xyz offset
    # Note: CenterOfMassOffset contains only the yz offsets, so we need to get
    # the x offset from XCoords
    x_B, x_A = binary_data["XCoords"]
    separation = x_A - x_B
    x_offset = x_A - target_params["MassB"] * separation
    y_offset, z_offset = binary_data["CenterOfMassOffset"]

    # Combine initial choices of free data in a dictionary
    initial_free_data = dict(
        conformal_mass_a=binary_data["ObjectRight"]["KerrSchild"]["Mass"],
        conformal_mass_b=binary_data["ObjectLeft"]["KerrSchild"]["Mass"],
        conformal_spin_a=binary_data["ObjectRight"]["KerrSchild"]["Spin"],
        conformal_spin_b=binary_data["ObjectLeft"]["KerrSchild"]["Spin"],
        center_of_mass_offset=[x_offset, y_offset, z_offset],
        linear_velocity=binary_data["LinearVelocity"],
        radial_expansion_velocity=binary_data["Expansion"],
        orbital_angular_velocity=binary_data["AngularVelocity"],
    )

    # File to write control diagnostic data
    data_file = open(f"{id_run_dir}/../ControlParamsData.txt", "w")

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
            control_run_dir = f"{id_run_dir}/../ControlParams_{iteration:03}"

            # Start with initial free data choices and update the ones being
            # controlled in `control_params` with the numeric value from `u`
            free_data = initial_free_data.copy()
            u_iterator = iter(u)
            for key in [FreeDataFromParams[param] for param in control_params]:
                if key in ScalarQuantities:
                    free_data[key] = next(u_iterator)
                else:
                    free_data[key] = [next(u_iterator) for _ in range(3)]

            # Run ID and find horizons
            generate_id(
                target_params,
                **free_data,
                separation=separation,
                run_dir=control_run_dir,
                control=False,
                evolve=False,
                scheduler=None,
                refinement_level=refinement_level,
                polynomial_order=polynomial_order,
                negative_expansion_bc=negative_expansion_bc,
            )

        # Initialize dictionary to hold the measured physical parameters
        measured_params: Dict[TargetParams, Union[float, Sequence[float]]] = {}

        # Get black hole physical parameters
        with spectre_h5.H5File(
            f"{control_run_dir}/Horizons.h5", "r"
        ) as horizons_file:
            AhA_quantities = to_dataframe(
                horizons_file.get_dat("AhA.dat")
            ).iloc[-1]

            if "MassA" in control_params:
                measured_params["MassA"] = AhA_quantities["ChristodoulouMass"]
            if "DimensionlessSpinA" in control_params:
                measured_params["DimensionlessSpinA"] = np.array(
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

            if "MassB" in control_params:
                measured_params["MassB"] = AhB_quantities["ChristodoulouMass"]
            if "DimensionlessSpinB" in control_params:
                measured_params["DimensionlessSpinB"] = np.array(
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

            if "CenterOfMass" in control_params:
                measured_params["CenterOfMass"] = np.array(
                    [
                        adm_integrals["CenterOfMass_x"],
                        adm_integrals["CenterOfMass_y"],
                        adm_integrals["CenterOfMass_z"],
                    ]
                )
            if "AdmLinearMomentum" in control_params:
                measured_params["AdmLinearMomentum"] = np.array(
                    [
                        adm_integrals["AdmLinearMomentum_x"],
                        adm_integrals["AdmLinearMomentum_y"],
                        adm_integrals["AdmLinearMomentum_z"],
                    ]
                )
            if "AdmMass" in control_params:
                measured_params["AdmMass"] = adm_integrals["AdmMass"]
            if "AdmAngularMomentumZ" in control_params:
                measured_params["AdmAngularMomentumZ"] = adm_integrals[
                    "AdmAngularMomentum_z"
                ]

        # Compute residual of physical parameters
        residual = np.array([])
        for key in control_params:
            target = target_params[key]
            assert target is not None, (
                f"Attempting to control parameter '{key}' but no target value"
                " is provided."
            )
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
    for key in [FreeDataFromParams[param] for param in control_params]:
        if key in ScalarQuantities:
            u = np.append(u, [initial_free_data[key]])
        else:
            u = np.append(u, initial_free_data[key])

    # Initial residual
    F = Residual(u)

    # Initialize Jacobian as an identity matrix
    J = np.identity(len(u))

    # Adjust non-unity components of the Jacobian
    q = target_params["MassRatio"]
    # The expression below is the reduced mass of the system, which shows up in
    # the Newtonian expressions further below.
    eta = q / (q + 1) ** 2
    param_index = 0
    for param in control_params:
        if param == "AdmMass":
            # The expression below comes from differentiating the Newtonian
            # approximation E_ADM ~ 1 + 1/2 eta adot0^2 D0^2, where adot0 is the
            # initial radial expansion velocity and D0 is the initial
            # separation.
            J[param_index, param_index] = (
                eta
                * initial_free_data["radial_expansion_velocity"]
                * separation**2
            )
        elif param == "AdmAngularMomentumZ":
            # The expression below comes from differentiating the Newtonian
            # approximation J_ADM ~ eta D0^2 Omega0, where D0 is the initial
            # separation and Omega0 is the initial angular orbital velocity.
            J[param_index, param_index] = eta * separation**2
        # Note: We can also set cross terms here to start with a more realistic
        # Jacobian.
        param_index += 1 if param in ScalarQuantities else 3

    # Indices of parameters for which the control is delayed in the first
    # iterations to avoid going off-bounds
    #
    # Note: We have experimented with other modifications to Broyden's
    # method, including damping the initial updates of the free data / Jacobian
    # and enforcing a diagonal Jacobian. None of them converged as fast as the
    # delay approach used here. When doing a more complete study in parameter
    # space, we should try to find a more robust approach that works for
    # multiple configurations.
    delayed_indices = np.array([], dtype=bool)
    delayed_params = [
        "CenterOfMass",
        "AdmLinearMomentum",
        "AdmMass",
        "AdmAngularMomentumZ",
    ]
    max_delayed_iteration = 1
    for key in control_params:
        if key in ScalarQuantities:
            delayed_indices = np.append(
                delayed_indices, [key in delayed_params]
            )
        else:
            delayed_indices = np.append(
                delayed_indices, [key in delayed_params] * 3
            )

    while iteration < max_iterations:
        iteration += 1

        # Update the free parameters using a quasi-Newton-Raphson method
        Delta_u = -np.dot(np.linalg.inv(J), F)
        if iteration <= max_delayed_iteration:
            Delta_u[delayed_indices] = 0.0
        u += Delta_u

        # Compute residual and check stopping condition
        F = Residual(u)
        if np.max(np.abs(F)) < residual_tolerance:
            break
        if iteration <= max_delayed_iteration:
            F[delayed_indices] = 0.0

        # Update the Jacobian using Broyden's method
        J += np.outer(F, Delta_u) / np.dot(Delta_u, Delta_u)

    data_file.close()

    return control_run_dir
