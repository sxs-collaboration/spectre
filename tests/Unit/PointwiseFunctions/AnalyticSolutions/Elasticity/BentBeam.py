# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from Elasticity.ConstitutiveRelations.IsotropicHomogeneous import (
    youngs_modulus, poisson_ratio)


def displacement(x, length, height, bending_moment, bulk_modulus,
                 shear_modulus):
    local_youngs_modulus = youngs_modulus(bulk_modulus, shear_modulus)
    local_poisson_ratio = poisson_ratio(bulk_modulus, shear_modulus)
    prefactor = 12. * bending_moment / (local_youngs_modulus * height**3)
    return np.array([
        -prefactor * x[0] * x[1], prefactor / 2. *
        (x[0]**2 + local_poisson_ratio * x[1]**2 - length**2 / 4.)
    ])


def strain(x, length, height, bending_moment, bulk_modulus, shear_modulus):
    local_youngs_modulus = youngs_modulus(bulk_modulus, shear_modulus)
    local_poisson_ratio = poisson_ratio(bulk_modulus, shear_modulus)
    prefactor = 12. * bending_moment / (local_youngs_modulus * height**3)
    result = np.zeros((2, 2))
    result[0, 0] = -prefactor * x[1]
    result[1, 1] = prefactor * local_poisson_ratio * x[1]
    return result


def stress(x, length, height, bending_moment, bulk_modulus, shear_modulus):
    return np.array([[12. * bending_moment / height**3 * x[1], 0], [0, 0]])


def source(x):
    return np.zeros(x.shape)
