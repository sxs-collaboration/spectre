# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def conformal_christoffel_second_kind(inverse_conformal_spatial_metric,
                                      field_d):
    return np.einsum("kl,ijl->kij", inverse_conformal_spatial_metric,
                     (np.einsum("ijl", field_d) + np.einsum("jil", field_d) -
                      np.einsum("lij", field_d)))
