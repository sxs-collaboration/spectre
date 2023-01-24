# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def adm_mass_integrand(field, alpha, beta):
    return 1. / (2. * np.pi) * beta * (alpha * (1. + field) + 1.)**(-7)
