# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def curved_fluxes(
    inv_spatial_metric, shift, lapse, conformal_factor, field_gradient
):
    metric_term = np.einsum("ij,j", inv_spatial_metric, field_gradient) / (
        conformal_factor
    ) ** (4)
    shift_term = (
        shift * np.einsum("i,i", shift, field_gradient) / (lapse) ** (2)
    )
    return metric_term - shift_term


def face_fluxes(
    inv_spatial_metric, shift, lapse, conformal_factor, face_normal, field
):
    metric_term = np.einsum("ij,j", inv_spatial_metric, face_normal) / (
        conformal_factor
    ) ** (4)
    shift_term = shift * np.einsum("i,i", shift, face_normal) / (lapse) ** (2)
    return (metric_term - shift_term) * field


def GB_source_term(
    epsilon2,
    epsilon4,
    weyl_electric_scalar,
    weyl_magnetic_scalar,
    field,
):
    gauss_bonnet_scalar = 8 * (weyl_electric_scalar - weyl_magnetic_scalar)
    coupling_function = (
        epsilon2 * field / 4 + epsilon4 * field * field * field / 4
    )
    return -coupling_function * gauss_bonnet_scalar


def linearized_GB_source_term(
    epsilon2,
    epsilon4,
    weyl_electric_scalar,
    weyl_magnetic_scalar,
    field,
    field_correction,
):
    gauss_bonnet_scalar = 8 * (weyl_electric_scalar - weyl_magnetic_scalar)
    linearized_coupling_function = (
        epsilon2 * field_correction / 4
        + 3 * epsilon4 * field * field * field_correction / 4
    )
    return -gauss_bonnet_scalar * linearized_coupling_function


def add_curved_sources(
    conformal_christoffel_contracted,
    field_flux,
    deriv_lapse,
    lapse,
    conformal_factor,
    conformal_factor_deriv,
):
    christoffel_term = -np.einsum(
        "i,i", conformal_christoffel_contracted, field_flux
    )
    lapse_term = -np.einsum("i,i", deriv_lapse, field_flux) / lapse
    conformal_factor_term = (
        -6
        * np.einsum("i,i", conformal_factor_deriv, field_flux)
        / conformal_factor
    )
    return christoffel_term + lapse_term + conformal_factor_term
