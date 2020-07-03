# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from Elasticity.ConstitutiveRelations.IsotropicHomogeneous import stress


def potential_energy_density(strain, coordinates, bulk_modulus, shear_modulus):
    local_stress = stress(strain, coordinates, bulk_modulus, shear_modulus)
    return -0.5 * np.einsum('ij, ij', strain, local_stress)
