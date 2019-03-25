# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Functions for testing Fluxes.cpp
def tilde_e_flux(tilde_e, tilde_s, tilde_p, lapse,
                   shift, spatial_metric, inv_spatial_metric):
    result = (lapse * (np.einsum("a, ia", tilde_s, inv_spatial_metric)) -
              shift * tilde_e)
    return result


def tilde_s_flux(tilde_e, tilde_s, tilde_p, lapse,
                 shift, spatial_metric, inv_spatial_metric):
    result = (lapse * (np.einsum("ia, aj", tilde_p, spatial_metric))-
              np.outer(shift, tilde_s))
    return result


# End of functions for testing Fluxes.cpp

