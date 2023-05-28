# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def adm_mass_integrand(field, alpha, beta):
    return 1.0 / (2.0 * np.pi) * beta * (alpha * (1.0 + field) + 1.0) ** (-7)
