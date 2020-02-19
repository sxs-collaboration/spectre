# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def constitutive_relation_2d(strain, bulk_modulus, shear_modulus):
    lame_constant = bulk_modulus - 2. / 3. * shear_modulus
    return -2. * shear_modulus * lame_constant / (
        lame_constant + 2. * shear_modulus
    ) * np.trace(strain) * np.eye(2) - 2. * shear_modulus * strain


def constitutive_relation_3d(strain, bulk_modulus, shear_modulus):
    lame_constant = bulk_modulus - 2. / 3. * shear_modulus
    return -2. * shear_modulus * strain - lame_constant * np.trace(
        strain) * np.eye(3)


def primal_fluxes_2d(strain, coordinates, bulk_modulus, shear_modulus):
    return -constitutive_relation_2d(strain, bulk_modulus, shear_modulus)


def primal_fluxes_3d(strain, coordinates, bulk_modulus, shear_modulus):
    return -constitutive_relation_3d(strain, bulk_modulus, shear_modulus)


def auxiliary_fluxes(field, dim):
    # Compute the tensor product with a Kronecker delta and symmetrize the last
    # two indices.
    tensor_product = np.tensordot(np.eye(dim), field, axes=0)
    return 0.5 * (tensor_product + np.transpose(tensor_product, (0, 2, 1)))


def auxiliary_fluxes_2d(field):
    return auxiliary_fluxes(field, 2)


def auxiliary_fluxes_3d(field):
    return auxiliary_fluxes(field, 3)
