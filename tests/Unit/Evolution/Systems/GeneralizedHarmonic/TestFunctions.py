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

def gauge_constraint(gauge_function, spacetime_normal_one_form,
                     spacetime_normal_vector, inverse_spatial_metric,
                     inverse_spacetime_metric, pi, phi):
    # Sums only go over spatial indices of second index of phi
    phi_ija = phi[:,1:,:]
    spacetime_normal_vector_I = spacetime_normal_vector[1:]
    constraint = gauge_function \
      + np.einsum("ij,ija->a", inverse_spatial_metric, phi_ija) \
      + np.einsum("b,ba->a", spacetime_normal_vector, pi) \
      - 0.5 * np.insert(np.einsum("bc,ibc->i", inverse_spacetime_metric, phi), \
                                  0, np.zeros(phi[0][0][0].shape)) \
      - 0.5 * np.einsum("a,i,bc,ibc->a", spacetime_normal_one_form, \
                        spacetime_normal_vector_I, \
                        inverse_spacetime_metric, phi) \
      - 0.5 * np.einsum("a,bc,bc->a", spacetime_normal_one_form, \
                        inverse_spacetime_metric, pi)
    return constraint
