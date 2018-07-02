# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Test functions for normal dot fluxes
def spacetime_metric_normal_dot_flux(spacetime_metric, pi, phi, gamma1, gamma2,
                                     lapse, shift, inverse_spatial_metric,
                                     unit_normal):
    return - (1. + gamma1) * np.dot(shift, unit_normal) * spacetime_metric


def pi_normal_dot_flux(spacetime_metric, pi, phi, gamma1, gamma2, lapse, shift,
                       inverse_spatial_metric, unit_normal):
    return - np.dot(shift, unit_normal) * pi  \
        + lapse * np.einsum("ki,k,iab->ab",
                            inverse_spatial_metric, unit_normal, phi) \
           - gamma1 * gamma2 * np.dot(shift, unit_normal) * spacetime_metric


def phi_dot_flux(spacetime_metric, pi, phi, gamma1, gamma2, lapse, shift,
                 inverse_spatial_metric, unit_normal):
    return - np.dot(shift, unit_normal) * phi \
        + lapse * np.einsum("i,ab->iab", unit_normal, pi) \
           - gamma2 * lapse * np.einsum("i,ab->iab", unit_normal,
                                        spacetime_metric)
