# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def strain_curved(deriv_displacement, metric, deriv_metric,
                  christoffel_first_kind, displacement):
    deriv_displacement_lo = (
        np.einsum('jk,ik->ij', metric, deriv_displacement) +
        np.einsum('k,ijk->ij', displacement, deriv_metric))
    return (0.5 *
            (deriv_displacement_lo + np.transpose(deriv_displacement_lo)) -
            np.einsum('kij,k', christoffel_first_kind, displacement))


def strain_flat(deriv_displacement):
    dim = len(deriv_displacement)
    return strain_curved(deriv_displacement,
                         metric=np.identity(dim),
                         deriv_metric=np.zeros((dim, dim, dim)),
                         christoffel_first_kind=np.zeros((dim, dim, dim)),
                         displacement=np.zeros((dim)))
