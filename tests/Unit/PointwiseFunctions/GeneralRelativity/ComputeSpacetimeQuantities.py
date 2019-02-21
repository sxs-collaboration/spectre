# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def spacetime_metric(lapse, shift, spatial_metric):
    dim = shift.size
    psi = np.zeros([dim + 1, dim + 1])
    psi[0, 0] = -lapse**2 + np.einsum("m,n,mn", shift, shift, spatial_metric)
    psi[1:, 0] = np.einsum("mi,m->i", spatial_metric, shift)
    psi[0, 1:] = psi[1:, 0]
    psi[1:, 1:] = spatial_metric
    return psi


def inverse_spacetime_metric(lapse, shift, inverse_spatial_metric):
    dim = shift.size
    inv_psi = np.zeros([dim + 1, dim + 1])
    inv_psi[0, 0] = -1. / lapse**2
    inv_psi[1:, 0] = shift / lapse**2
    inv_psi[0, 1:] = inv_psi[1:, 0]
    inv_psi[1:,
            1:] = inverse_spatial_metric - np.outer(shift, shift) / lapse**2
    return inv_psi


def dt_spacetime_metric(lapse, dt_lapse, shift, dt_shift, spatial_metric,
                        dt_spatial_metric):
    dim = shift.size
    dt_psi = np.zeros([dim + 1, dim + 1])
    dt_psi[0, 0] = - 2 * lapse * dt_lapse \
        + 2 * np.einsum("mn,m,n", spatial_metric, shift, dt_shift) \
        + np.einsum("m,n,mn", shift, shift, dt_spatial_metric)
    dt_psi[1:, 0] = np.einsum("mi,m", spatial_metric, dt_shift) \
        + np.einsum("m,mi", shift, dt_spatial_metric)
    dt_psi[1:, 1:] = dt_spatial_metric
    dt_psi[0, 1:] = dt_psi[1:, 0]  # Symmetrise
    return dt_psi


def spatial_deriv_spacetime_metric(lapse, deriv_lapse, shift, deriv_shift,
                                   spatial_metric, deriv_spatial_metric):
    dim = shift.size
    deriv_psi = np.zeros([dim, dim + 1, dim + 1])
    deriv_psi[:, 0, 0] = - 2. * lapse * deriv_lapse \
        + 2. * np.einsum("mn,m,kn->k",
                         spatial_metric, shift, deriv_shift) \
        + np.einsum("m,n,kmn->k",
                    shift, shift, deriv_spatial_metric)
    deriv_psi[:, 1:, 0] = np.einsum("mi,km->ki", spatial_metric, deriv_shift) \
        + np.einsum("m,kmi->ki", shift, deriv_spatial_metric)
    deriv_psi[:, 1:, 1:] = deriv_spatial_metric
    deriv_psi[:, 0, 1:] = deriv_psi[:, 1:, 0]  # Symmetrise
    return deriv_psi


def derivatives_of_spacetime_metric(lapse, dt_lapse, deriv_lapse, shift,
                                    dt_shift, deriv_shift, spatial_metric,
                                    dt_spatial_metric, deriv_spatial_metric):
    dim = shift.size
    d4_psi = np.zeros([dim + 1, dim + 1, dim + 1])
    # Spatial derivatives
    d4_psi[0, :, :] = dt_spacetime_metric(lapse, dt_lapse, shift, dt_shift,
                                          spatial_metric, dt_spatial_metric)
    d4_psi[1:, :, :] = spatial_deriv_spacetime_metric(
        lapse, deriv_lapse, shift, deriv_shift, spatial_metric,
        deriv_spatial_metric)
    return d4_psi


def spacetime_normal_vector(lapse, shift):
    dim = shift.size
    vector = np.zeros([dim + 1])
    vector[0] = 1. / lapse
    vector[1:] = -shift / lapse
    return vector


def extrinsic_curvature(lapse, shift, deriv_shift, spatial_metric,
                        dt_spatial_metric, deriv_spatial_metric):
    ext_curve = np.einsum("k,kij", shift, deriv_spatial_metric) \
        + np.einsum("ki,jk", spatial_metric, deriv_shift) \
        + np.einsum("kj,ik", spatial_metric, deriv_shift) \
        - dt_spatial_metric
    ext_curve *= 0.5 / lapse
    return ext_curve
