# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def DDKG_normal_normal_projection(
    lapse,
    shift,
    inverse_spatial_metric,
    phi_scalar,
    d_pi_scalar,
    dt_pi_scalar,
    d_lapse,
):
    lie_pi = dt_pi_scalar - np.dot(shift, d_pi_scalar)
    lie_pi *= 1.0 / lapse

    acc_dot_phi = np.transpose(phi_scalar) @ inverse_spatial_metric @ d_lapse
    acc_dot_phi *= 1.0 / lapse

    result = -lie_pi - acc_dot_phi

    return result


def DDKG_normal_spatial_projection(
    inverse_spatial_metric, extrinsic_curvature, phi_scalar, d_pi_scalar
):
    result = (
        extrinsic_curvature @ inverse_spatial_metric @ phi_scalar
    ) - d_pi_scalar

    return result


def DDKG_spatial_spatial_projection(
    extrinsic_curvature,
    spatial_christoffel_second_kind,
    pi_scalar,
    phi_scalar,
    d_phi_scalar,
):
    chris_times_phi = np.einsum(
        "kij, k->ij", spatial_christoffel_second_kind, phi_scalar
    )

    # Random number test does not impose symmetry of d_phi
    # D_phi = d_phi_scalar - chris_times_phi

    # Symmetrizing
    D_phi = 0.5 * (d_phi_scalar + np.transpose(d_phi_scalar)) - chris_times_phi

    result = D_phi - pi_scalar * extrinsic_curvature

    return result
