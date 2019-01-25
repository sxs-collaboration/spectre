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


def deriv_lapse(lapse, spacetime_unit_normal, phi):
    t1 = np.einsum('iab,b->ia', phi, spacetime_unit_normal)
    t1 = np.einsum('ia,a->i', t1, spacetime_unit_normal)
    return -0.5 * lapse * t1


def dt_lapse(lapse, shift, spacetime_unit_normal, phi, pi):
    t1 = np.einsum('ab,b->a', pi, spacetime_unit_normal)
    t1 = np.einsum('a,a', t1, spacetime_unit_normal)
    t1 *= lapse
    t2 = np.einsum('iab,b->ia', phi, spacetime_unit_normal)
    t2 = np.einsum('ia,a->i', t2, spacetime_unit_normal)
    t2 = np.einsum('i,i', t2, shift)
    return 0.5 * lapse * (t1 - t2)


def deriv_spatial_metric(phi):
    return phi[:,1:,1:]


def dt_spatial_metric(lapse, shift, phi, pi):
    return (-lapse * pi + np.einsum('k,kab->ab', shift, phi))[1:,1:]


def spacetime_deriv_detg(sqrt_det_spatial_metric, inverse_spatial_metric,
        dt_spatial_metric, phi):
    det_spatial_metric = sqrt_det_spatial_metric**2
    deriv_of_g = deriv_spatial_metric(phi)
    dtg = np.einsum('jk,jk', inverse_spatial_metric, dt_spatial_metric)
    dtg *= det_spatial_metric
    dxg = np.einsum('jk,ijk->i', inverse_spatial_metric, deriv_of_g)
    dxg *= det_spatial_metric
    dg     = np.zeros(1  + len(dxg))
    dg[0]  = dtg
    dg[1:] = dxg
    return dg


# End tests for Test_ComputeGhQuantities.cpp

# Begin tests for Test_Ricci.cpp


def ricci_tensor(christoffel, deriv_christoffel):
    return (np.einsum("ccab", deriv_christoffel) - 0.5 * (np.einsum(
        "bcac", deriv_christoffel) + np.einsum("acbc", deriv_christoffel)) +
            np.einsum("dab,ccd", christoffel, christoffel) -
            np.einsum("dac,cbd", christoffel, christoffel))


# End tests for Test_Ricci.cpp

# Begin tests for Test_KerrSchildCoords.cpp


def ks_coords_r_squared(coords, bh_mass, bh_dimless_spin):
    a_squared = (bh_mass * bh_dimless_spin)**2
    temp = 0.5 * (coords[0] * coords[0] + coords[1] * coords[1] +
                  coords[2] * coords[2] - a_squared)
    return temp + np.sqrt(temp * temp + a_squared * coords[2] * coords[2])


def ks_coords_jacobian(coords, bh_mass, bh_dimless_spin):
    result = np.zeros((3, 3))
    spin_a = bh_mass * bh_dimless_spin
    a_squared = spin_a**2
    r_squared = ks_coords_r_squared(coords, bh_mass, bh_dimless_spin)
    r = np.sqrt(r_squared)
    sin_theta = np.sqrt((coords[0]**2 + coords[1]**2) /
                        (r_squared + a_squared))
    cos_theta = coords[2] / r
    inv_denom = 1.0 / np.sqrt((coords[0]**2 + coords[1]**2) *
                          (r_squared + a_squared))
    sin_phi = (coords[1] * r - spin_a * coords[0]) * inv_denom
    cos_phi = (coords[0] * r + spin_a * coords[1]) * inv_denom

    result[0, 0] = sin_theta * cos_phi
    result[0, 1] = (r * cos_phi - spin_a * sin_phi) * cos_theta
    result[0, 2] = -(r * sin_phi + spin_a * cos_phi) * sin_theta
    result[1, 0] = sin_theta * sin_phi
    result[1, 1] = (r * sin_phi + spin_a * cos_phi) * cos_theta
    result[1, 2] = (r * cos_phi - spin_a * sin_phi) * sin_theta
    result[2, 0] = cos_theta
    result[2, 1] = -r * sin_theta
    result[2, 2] = 0.0
    return result


def ks_coords_cartesian_from_spherical_ks(vector, cartesian_coords, bh_mass,
                                          bh_dimless_spin):
    return np.einsum("ij,j",
                     ks_coords_jacobian(cartesian_coords, bh_mass,
                                        bh_dimless_spin), vector)


# End tests for Test_KerrSchildCoords.cpp

# Begin tests for Test_IndexManipulation.cpp


def raise_or_lower_first_index(tensor, metric):
    return np.einsum("ij,ikl", metric, tensor)


def trace_last_indices(tensor, metric):
    return np.einsum("ij,kij", metric, tensor)


# Endtests for Test_IndexManipulation.cpp
