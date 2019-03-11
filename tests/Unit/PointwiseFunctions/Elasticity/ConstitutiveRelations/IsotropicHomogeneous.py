# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def stress(strain, x, bulk_modulus, shear_modulus):
    dim = len(x)
    krond = np.eye(dim)
    strain_trace = np.trace(strain)
    traceless_strain = strain - strain_trace / dim * krond
    if dim == 3:
        trace_factor = bulk_modulus
    elif dim == 2:
        trace_factor = 9 * bulk_modulus * shear_modulus \
            / (3 * bulk_modulus + 4 * shear_modulus)
    else:
        raise NotImplementedError
    return -trace_factor * strain_trace * krond \
        - 2 * shear_modulus * traceless_strain
