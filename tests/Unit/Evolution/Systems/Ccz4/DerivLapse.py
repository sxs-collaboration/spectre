# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def grad_grad_lapse(lapse, christoffel_second_kind, field_a, d_field_a):
    return (lapse * np.einsum("i,j", field_a, field_a) -
            lapse * np.einsum("kij,k", christoffel_second_kind, field_a) +
            0.5 * lapse *
            (np.einsum("ij", d_field_a) + np.einsum("ij->ji", d_field_a)))


def divergence_lapse(conformal_factor_squared, inverse_conformal_metric,
                     grad_grad_lapse):
    return (conformal_factor_squared *
            np.einsum("ij,ij", inverse_conformal_metric, grad_grad_lapse))
