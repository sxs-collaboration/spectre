# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

import spectre.IO.H5 as spectre_h5


def write_mock_trajectory_data(
    reductions_filename: str,
    t: np.ndarray,
    separation: float,
):
    """Generate mock-up inspiral data

    Simple circular Newtonian orbits for an equal-mass binary. Can be expanded
    to more general orbits if needed.
    """
    omega = np.sqrt(1.0 / (separation**3))
    phase = omega * t
    centers_a = np.array(
        [
            0.5 * separation * np.cos(phase),
            0.5 * separation * np.sin(phase),
            0.0 * phase,
        ]
    )
    centers_b = np.array(
        [
            -0.5 * separation * np.cos(phase),
            -0.5 * separation * np.sin(phase),
            0.0 * phase,
        ]
    )

    # Write to H5 file
    with spectre_h5.H5File(reductions_filename, "a") as h5_file:
        for ab, centers in zip("AB", [centers_a, centers_b]):
            datfile = h5_file.insert_dat(
                f"ApparentHorizons/ControlSystemAh{ab}_Centers.dat",
                legend=[
                    "Time",
                    "GridCenter_x",
                    "GridCenter_y",
                    "GridCenter_z",
                    "InertialCenter_x",
                    "InertialCenter_y",
                    "InertialCenter_z",
                ],
                version=0,
            )
            for t_i, x_i in zip(t, centers.T):
                datfile.append([t_i, *x_i, *x_i])
            h5_file.close_current_object()
