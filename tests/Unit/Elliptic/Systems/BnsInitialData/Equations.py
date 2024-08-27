# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def potential_fluxes(
    rotational_shift_stress, inv_spatial_metric, auxiliary_velocity
):
    return np.einsum(
        "ij,j", inv_spatial_metric - rotational_shift_stress, auxiliary_velocity
    )


def fluxes_on_face(
    face_normal, face_normal_vector, rotational_shift_stress, velocity_potential
):
    return velocity_potential * (
        face_normal_vector
        - np.einsum("ij,j", rotational_shift_stress, face_normal)
    )


def add_potential_sources(
    log_deriv_lapse_over_specific_enthalpy,
    christoffel_contracted,
    flux_for_potential,
):
    return -np.einsum(
        "i,i",
        christoffel_contracted + log_deriv_lapse_over_specific_enthalpy,
        flux_for_potential,
    )
