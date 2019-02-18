# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Functions for testing Sources.cpp
def source_tilde_e(tilde_e,tilde_s,tilde_p,lapse,
                   d_lapse,d_shift,d_spatial_metric,
                   inv_spatial_metric,extrinsic_curvature):
    result = (lapse * np.einsum("ab, ab",tilde_p,extrinsic_curvature) -
              np.einsum("ab, ab", inv_spatial_metric,
                       np.outer(tilde_s, d_lapse)))
    return result


def source_tilde_s(tilde_e,tilde_s,tilde_p,lapse,
                   d_lapse,d_shift,d_spatial_metric,
                   inv_spatial_metric,extrinsic_curvature):
    result = (0.5 * lapse *
              np.einsum("ab, iab", tilde_p, d_spatial_metric) +
              np.einsum("a, ia", tilde_s, d_shift) -
              tilde_e * d_lapse)
    return result


# End of functions for testing Sources.cpp

