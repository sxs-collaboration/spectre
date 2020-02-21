# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def displacement(x, length, height, bending_moment, bulk_modulus,
                 shear_modulus):
    youngs_modulus = 9. * bulk_modulus * shear_modulus / \
        (3. * bulk_modulus + shear_modulus)
    poisson_ratio = (3. * bulk_modulus - 2. * shear_modulus) / \
        (6. * bulk_modulus + 2. * shear_modulus)
    prefactor = 12. * bending_moment / (youngs_modulus * height**3)
    return np.array([
        -prefactor * x[0] * x[1],
        prefactor / 2. * (x[0]**2 + poisson_ratio * x[1]**2 - length**2 / 4.)
    ])


def stress(x, length, height, bending_moment, bulk_modulus, shear_modulus):
    return np.array([[12. * bending_moment / height**3 * x[1], 0], [0, 0]])


def source(x):
    return np.zeros(x.shape)
