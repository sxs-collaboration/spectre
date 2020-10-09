# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def flat_cartesian_fluxes(field_gradient):
    return field_gradient


def curved_fluxes(inv_spatial_metric, field_gradient):
    return np.einsum('ij,j', inv_spatial_metric, field_gradient)


def add_curved_sources(christoffel_contracted, field_flux):
    return -np.einsum('i,i', christoffel_contracted, field_flux)


def auxiliary_fluxes(field, dim):
    return np.diag(np.repeat(field, dim))


def auxiliary_fluxes_1d(field):
    return auxiliary_fluxes(field, 1)


def auxiliary_fluxes_2d(field):
    return auxiliary_fluxes(field, 2)


def auxiliary_fluxes_3d(field):
    return auxiliary_fluxes(field, 3)
