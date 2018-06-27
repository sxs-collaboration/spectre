# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def christoffel_first_kind(d_metric):
    dim = d_metric.shape[0]
    return 0.5 * np.array([[[d_metric[b, c, a] + d_metric[a, c, b] - d_metric[
        c, a, b] for b in range(dim)] for a in range(dim)] for c in range(dim)])


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


# Begin tests for Test_ComputeSpacetimeQuantities.cpp


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


# End tests for Test_ComputeSpacetimeQuantities.cpp

# Begin tests for Test_ComputeGhQuantities.cpp


def gh_pi(lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric,
          phi):
    return (np.einsum("iab,i", phi, shift) - dt_spacetime_metric(
        lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric)
            ) / lapse


def gh_gauge_source(lapse, dt_lapse, deriv_lapse, shift, dt_shift, deriv_shift,
                    spatial_metric, tr_extrinsic_curvature,
                    trace_christoffel_last_indices):
    dim = shift.size
    source = np.zeros(dim + 1)
    shift_dot_d_shift = np.einsum("k,ki", shift, deriv_shift)
    inv_lapse = 1. / lapse
    source[1:] = inv_lapse ** 2 * np.einsum("ij,j", spatial_metric,
                                            dt_shift - shift_dot_d_shift) \
                 + deriv_lapse / lapse - trace_christoffel_last_indices
    source[0] = - dt_lapse * inv_lapse \
                + inv_lapse * np.dot(shift, deriv_lapse) \
                + np.dot(shift, source[1:]) - lapse * tr_extrinsic_curvature
    return source


# End tests for Test_ComputeGhQuantities.cpp

# Begin tests for Test_Ricci.cpp


def ricci_tensor(christoffel, deriv_christoffel):
    return (np.einsum("ccab", deriv_christoffel) - 0.5 * (np.einsum(
        "bcac", deriv_christoffel) + np.einsum("acbc", deriv_christoffel)) +
            np.einsum("dab,ccd", christoffel, christoffel) -
            np.einsum("dac,cbd", christoffel, christoffel))


# End tests for Test_Ricci.cpp

# Begin tests for Test_IndexManipulation.cpp


def raise_or_lower_first_index(tensor, metric):
    return np.einsum("ij,ikl", metric, tensor)


def trace_last_indices(tensor, metric):
    return np.einsum("ij,kij", metric, tensor)


# Endtests for Test_IndexManipulation.cpp
