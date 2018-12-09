# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# Curved-space cross product returning a vector, given vectors a, b
def cross_product_up(a, b, inverse_metric, det_metric):
    return np.einsum('ij,j', inverse_metric, np.cross(a, b)) * \
        np.sqrt(det_metric)

# Curved-space cross product returning a covector, given vector a, covector b
def cross_product_lo(a, b, inverse_metric, det_metric):
    return np.cross(a, np.einsum('ij,j', inverse_metric, b)) * \
        np.sqrt(det_metric)
