# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def mass_source(psi, mass_psi):
    return mass_psi * mass_psi * psi


def add_scalar_source_to_dt_pi_scalar(scalar_source, lapse):
    return 0.1234 + lapse * scalar_source
