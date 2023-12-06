# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def flat_cartesian_fluxes(field_gradient):
    return field_gradient


def curved_fluxes(inv_spatial_metric, field_gradient):
    return np.einsum("ij,j", inv_spatial_metric, field_gradient)


def add_curved_sources(christoffel_contracted, field_flux):
    return -np.einsum("i,i", christoffel_contracted, field_flux)
