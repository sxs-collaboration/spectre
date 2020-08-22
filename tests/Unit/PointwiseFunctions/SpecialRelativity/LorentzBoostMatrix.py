# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def lorentz_boost_matrix(velocity):
    boost_matrix = np.zeros((velocity.size + 1, velocity.size + 1))
    velocity_squared = np.einsum("i,i->", velocity, velocity)
    lorentz_factor = 1.0 / np.sqrt(1.0 - velocity_squared)

    prefactor = lorentz_factor / (1 + np.sqrt(1.0 - velocity_squared))

    boost_matrix.itemset((0, 0), lorentz_factor)

    for i in np.arange(0, velocity.size, 1):
        boost_matrix.itemset((0, i + 1), lorentz_factor * velocity[i])
        boost_matrix.itemset((i + 1, 0), lorentz_factor * velocity[i])
        for j in np.arange(0, velocity.size, 1):
            if (i == j):
                boost_matrix.itemset(
                    (i + 1, j + 1),
                    (velocity[i] * velocity[j] * prefactor) + 1.0)
            else:
                boost_matrix.itemset((i + 1, j + 1),
                                     (velocity[i] * velocity[j] * prefactor))

    return boost_matrix
