# Distributed under the MIT License.
# See LICENSE.txt for details.
import numpy as np


def rotational_shift(shift, spatial_rotational_killing_vector):
    return shift + spatial_rotational_killing_vector


def rotational_shift_stress(rotational_shift, lapse, spatial_metric):
    return (
        np.outer(
            rotational_shift,
            np.einsum("ij, i", spatial_metric, rotational_shift),
        )
        / lapse**2
    )


def derivative_rotational_shift_over_lapse(
    rotational_shift,
    deriv_of_shift,
    lapse,
    deriv_of_lapse,
    deriv_of_spatial_rotational_killing_vector,
):
    return (
        deriv_of_shift + deriv_of_spatial_rotational_killing_vector
    ) / lapse - np.outer(deriv_of_lapse / lapse**2, rotational_shift)


def divergence_rotational_shift_stress(
    rotational_shift, deriv_rotational_shift_over_lapse, lapse, spatial_metric
):
    return np.einsum(
        "ij, i",
        spatial_metric,
        np.einsum(
            "i, ij", rotational_shift / lapse, deriv_rotational_shift_over_lapse
        )
        + np.einsum("ii", deriv_rotational_shift_over_lapse)
        * rotational_shift
        / lapse,
    )


def enthalpy_density_squared(
    rotational_shift,
    lapse,
    velocity_potential_gradient,
    inverse_spatial_metric,
    euler_enthalpy_constant,
):
    return 1 / lapse**2 * (
        euler_enthalpy_constant
        + np.einsum("i, i", rotational_shift, velocity_potential_gradient)
    ) ** 2 - np.einsum(
        "i, i",
        velocity_potential_gradient,
        np.einsum("ij, j", inverse_spatial_metric, velocity_potential_gradient),
    )


def spatial_rotational_killing_vector(
    x, local_angular_velocity_around_z, sqrt_det_spatial_metric
):
    print(
        np.cross(
            local_angular_velocity_around_z
            * np.array(
                [np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0])]
            ),
            x,
        )
    )
    return sqrt_det_spatial_metric * np.cross(
        local_angular_velocity_around_z
        * np.array(
            [np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0])]
        ),
        x,
    )


def derivative_spatial_rotational_killing_vector(
    x, local_angular_velocity_around_z, sqrt_det_spatial_metric
):
    return np.zeros((3, len(x)))
