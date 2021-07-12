# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

from .ComputeSpacetimeQuantities import (spatial_deriv_spacetime_metric,
                                         dt_spacetime_metric)


def phi(lapse, deriv_lapse, shift, deriv_shift, spatial_metric,
        deriv_spatial_metric):
    return spatial_deriv_spacetime_metric(lapse, deriv_lapse, shift,
                                          deriv_shift, spatial_metric,
                                          deriv_spatial_metric)


def pi(lapse, dt_lapse, shift, dt_shift, spatial_metric, dt_spatial_metric,
       phi):
    return (np.einsum("iab,i", phi, shift) -
            dt_spacetime_metric(lapse, dt_lapse, shift, dt_shift,
                                spatial_metric, dt_spatial_metric)) / lapse


def gauge_source(lapse, dt_lapse, deriv_lapse, shift, dt_shift, deriv_shift,
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


def deriv_shift(lapse, inverse_spacetime_metric, spacetime_unit_normal, phi):
    t1 = np.einsum('b,iab->ia', spacetime_unit_normal, phi)
    t1 = np.einsum('ja,ia->ij', inverse_spacetime_metric[1:, :], t1) * lapse
    t2 = np.einsum('a,iab->ib', spacetime_unit_normal, phi)
    t2 = np.einsum('b,ib->i', spacetime_unit_normal, t2)
    t2 = np.einsum('i,j->ij', t2, spacetime_unit_normal[1:]) * lapse
    return t1 + t2


def dt_shift(lapse, shift, inverse_spatial_metric, spacetime_unit_normal, phi,
             pi):
    t1 = np.einsum('a,ja->j', spacetime_unit_normal, pi[1:, :])
    t1 = np.einsum('j,ij->i', t1, inverse_spatial_metric)
    t2 = np.einsum('jka,a->jk', phi[:, 1:, :], spacetime_unit_normal)
    t2 = np.einsum('jk,ik->ji', t2, inverse_spatial_metric)
    t2 = np.einsum('ji,j->i', t2, shift)
    return lapse * (t2 - lapse * t1)


def dt_lower_shift(lapse, shift, spatial_metric, spacetime_unit_normal, phi,
                   pi):
    inverse_spatial_metric = np.linalg.inv(spatial_metric)
    dt_shift_ = dt_shift(lapse, shift, inverse_spatial_metric,
                         spacetime_unit_normal, phi, pi)
    t1 = np.einsum('ij,j->i', spatial_metric, dt_shift_)
    dt_spatial_metric =\
        (-lapse * pi + np.einsum('k,kab->ab', shift, phi))[1:, 1:]
    t2 = np.einsum('j,ij->i', shift, dt_spatial_metric)
    return t1 + t2


def spacetime_deriv_norm_shift(lapse, shift, spatial_metric,
                               inverse_spatial_metric,
                               inverse_spacetime_metric, spacetime_unit_normal,
                               phi, pi):
    lower_shift = np.einsum('ij,j->i', spatial_metric, shift)
    deriv_shift_ = deriv_shift(lapse, inverse_spacetime_metric,
                               spacetime_unit_normal, phi)
    dt_lower_shift_ = dt_lower_shift(lapse, shift, spatial_metric,
                                     spacetime_unit_normal, phi, pi)
    dt_shift_ = dt_shift(lapse, shift, inverse_spatial_metric,
                         spacetime_unit_normal, phi, pi)
    t0 = np.einsum('i,i', lower_shift, dt_shift_) \
        + np.einsum('i,i', shift, dt_lower_shift_)
    ti = np.einsum('i,ji->j', lower_shift, deriv_shift_) \
        + np.einsum('i,ji->j', shift, phi[:, 0, 1:])
    ta = np.zeros(1 + len(ti))
    ta[0] = t0
    ta[1:] = ti
    return ta


def deriv_spatial_metric(phi):
    return phi[:, 1:, 1:]


def dt_spatial_metric(lapse, shift, phi, pi):
    return (-lapse * pi + np.einsum('k,kab->ab', shift, phi))[1:, 1:]


def gh_dt_spacetime_metric(lapse, shift, pi, phi):
    return (-lapse * pi + np.einsum('k,kab', shift, phi))


def spacetime_deriv_detg(sqrt_det_spatial_metric, inverse_spatial_metric,
                         dt_spatial_metric, phi):
    det_spatial_metric = sqrt_det_spatial_metric**2
    deriv_of_g = deriv_spatial_metric(phi)
    dtg = np.einsum('jk,jk', inverse_spatial_metric, dt_spatial_metric)
    dtg *= det_spatial_metric
    dxg = np.einsum('jk,ijk->i', inverse_spatial_metric, deriv_of_g)
    dxg *= det_spatial_metric
    dg = np.zeros(1 + len(dxg))
    dg[0] = dtg
    dg[1:] = dxg
    return dg


def extrinsic_curvature(spacetime_normal_vector, pi, phi):
    return (
        0.5 * pi[1:, 1:] +
        0.5 * np.einsum('ija,a->ij', phi[:, 1:, :], spacetime_normal_vector) +
        0.5 * np.einsum('jia,a->ij', phi[:, 1:, :], spacetime_normal_vector))


def covariant_deriv_extrinsic_curvture(extrinsic_curvature,
                                       spacetime_unit_nomal_vector,
                                       spatial_christoffel_second_kind,
                                       inverse_spacetime_metric, phi, d_pi,
                                       d_phi):
    term4 = np.einsum('ija,b,ca,kcb->kij', phi[:, 1:, :],
                      spacetime_unit_nomal_vector, inverse_spacetime_metric,
                      phi)
    term5 = np.einsum('jia,b,ca,kcb->kij', phi[:, 1:, :],
                      spacetime_unit_nomal_vector, inverse_spacetime_metric,
                      phi)
    term6 = np.einsum('ija,b,c,a,kcb->kij', phi[:, 1:, :],
                      spacetime_unit_nomal_vector,
                      0.5 * spacetime_unit_nomal_vector,
                      spacetime_unit_nomal_vector, phi)
    term7 = np.einsum('jia,b,c,a,kcb->kij', phi[:, 1:, :],
                      spacetime_unit_nomal_vector,
                      0.5 * spacetime_unit_nomal_vector,
                      spacetime_unit_nomal_vector, phi)

    cdk = d_pi[:, 1:, 1:] + np.einsum(
        'kija,a->kij',
        d_phi[:, :, 1:, :], spacetime_unit_nomal_vector) + np.einsum(
            'kjia,a->kij', d_phi[:, :, 1:, :],
            spacetime_unit_nomal_vector) - term4 - term5 - term6 - term7

    return 0.5 * cdk - np.einsum(
        'lik,lj->kij',
        spatial_christoffel_second_kind, extrinsic_curvature) - np.einsum(
            'ljk,li->kij', spatial_christoffel_second_kind,
            extrinsic_curvature)


def gh_spatial_ricci_tensor(phi, deriv_phi, inverse_spatial_metric):
    ricci_deriv_terms = 0.25 * (
        np.einsum("kl,jlki->ij", inverse_spatial_metric, deriv_phi[:, :, 1:,
                                                                   1:]) +
        np.einsum("kl,ilkj->ij", inverse_spatial_metric, deriv_phi[:, :, 1:,
                                                                   1:]) -
        np.einsum("kl,jikl->ij", inverse_spatial_metric, deriv_phi[:, :, 1:,
                                                                   1:]) -
        np.einsum("kl,ijkl->ij", inverse_spatial_metric, deriv_phi[:, :, 1:,
                                                                   1:]) +
        np.einsum("kl,kijl->ij", inverse_spatial_metric, deriv_phi[:, :, 1:,
                                                                   1:]) +
        np.einsum("kl,kjil->ij", inverse_spatial_metric, deriv_phi[:, :, 1:,
                                                                   1:]) -
        2 * np.einsum("kl,lkij->ij", inverse_spatial_metric,
                      deriv_phi[:, :, 1:, 1:]))

    spatial_phi_Ijj = 0.5 * np.einsum("kl,lij->kij", inverse_spatial_metric,
                                      phi[:, 1:, 1:])
    spatial_phi_ijK = 0.5 * np.einsum("kl,ijl->ijk", inverse_spatial_metric,
                                      phi[:, 1:, 1:])
    d_minus_two_b = np.einsum(
        "kl,lii->k", inverse_spatial_metric, spatial_phi_ijK) - 2 * np.einsum(
            "kl,iil->k", inverse_spatial_metric, spatial_phi_Ijj)

    return ricci_deriv_terms + 0.5 * (
        np.einsum("ijk,k->ij", phi[:, 1:, 1:], d_minus_two_b) +
        np.einsum("jik,k->ij", phi[:, 1:, 1:], d_minus_two_b) -
        np.einsum("kij,k->ij", phi[:, 1:, 1:], d_minus_two_b)) + np.einsum(
            "ikl,jlk->ij", spatial_phi_ijK, spatial_phi_ijK) + 2 * np.einsum(
                "kil,kjl->ij", spatial_phi_Ijj, spatial_phi_ijK
            ) - 2 * np.einsum("kli,lkj->ij", spatial_phi_Ijj, spatial_phi_Ijj)
