# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def constitutive_relation_2d(strain, bulk_modulus, shear_modulus):
    lame_constant = bulk_modulus - 2.0 / 3.0 * shear_modulus
    return (
        -2.0
        * shear_modulus
        * lame_constant
        / (lame_constant + 2.0 * shear_modulus)
        * np.trace(strain)
        * np.eye(2)
        - 2.0 * shear_modulus * strain
    )


def constitutive_relation_3d(strain, bulk_modulus, shear_modulus):
    lame_constant = bulk_modulus - 2.0 / 3.0 * shear_modulus
    return -2.0 * shear_modulus * strain - lame_constant * np.trace(
        strain
    ) * np.eye(3)


def primal_fluxes_2d(
    deriv_displacement, coordinates, bulk_modulus, shear_modulus
):
    strain = 0.5 * (deriv_displacement + deriv_displacement.T)
    return -constitutive_relation_2d(strain, bulk_modulus, shear_modulus)


def primal_fluxes_3d(
    deriv_displacement, coordinates, bulk_modulus, shear_modulus
):
    strain = 0.5 * (deriv_displacement + deriv_displacement.T)
    return -constitutive_relation_3d(strain, bulk_modulus, shear_modulus)


def add_curved_sources(christoffel_second_kind, christoffel_contracted, stress):
    return -np.einsum("i,ij", christoffel_contracted, stress) - np.einsum(
        "ijk,jk", christoffel_second_kind, stress
    )
