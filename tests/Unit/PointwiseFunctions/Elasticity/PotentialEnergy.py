# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def potential_energy_2d(strain, bulk_modulus, shear_modulus):
    stress = constitutive_relation_2d(strain, bulk_modulus, shear_modulus)
    return -0.5 * np.einsum('ij, ij', strain, stress)


def potential_energy_3d(strain, bulk_modulus, shear_modulus):
    stress = constitutive_relation_3d(strain, bulk_modulus, shear_modulus)
    return -0.5 * np.einsum('ij, ij', strain, stress)


def potential_energy(strain, coordinates, bulk_modulus, shear_modulus):
    dim = coordinates.shape[0]
    if dim == 2:
        return potential_energy_2d(strain, bulk_modulus, shear_modulus)
    elif dim == 3:
        return potential_energy_3d(strain, bulk_modulus, shear_modulus)


def constitutive_relation_2d(strain, bulk_modulus, shear_modulus):
    lame_constant = bulk_modulus - 2. / 3. * shear_modulus
    return -2. * shear_modulus * lame_constant / (
        lame_constant + 2. * shear_modulus
    ) * np.trace(strain) * np.eye(2) - 2. * shear_modulus * strain


def constitutive_relation_3d(strain, bulk_modulus, shear_modulus):
    lame_constant = bulk_modulus - 2. / 3. * shear_modulus
    return -2. * shear_modulus * strain - lame_constant * np.trace(
        strain) * np.eye(3)
