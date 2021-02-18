# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def longitudinal_operator(strain, inv_metric):
    projection = (np.einsum('ij,kl->ikjl', inv_metric, inv_metric) -
                  1. / 3. * np.einsum('ij,kl->ijkl', inv_metric, inv_metric))
    return 2. * np.einsum('ijkl,kl', projection, strain)


def longitudinal_operator_flat_cartesian(strain):
    return longitudinal_operator(strain, np.identity(3))
