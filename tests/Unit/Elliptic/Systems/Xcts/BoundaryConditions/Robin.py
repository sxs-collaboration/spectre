# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from numpy import abs, sqrt
from PointwiseFunctions.Xcts.LongitudinalOperator import (
    longitudinal_operator_flat_cartesian,
)


def robin_boundary_condition_scalar(field, x):
    r = np.linalg.norm(x)
    return -field / r


def robin_boundary_condition_shift(shift, deriv_shift, x, face_normal):
    proj = np.eye(3) - np.outer(face_normal, face_normal)
    r = np.linalg.norm(x)
    deriv_shift = (
        np.einsum("ik,kj->ij", proj, deriv_shift)
        - np.outer(face_normal, shift) / r
    )
    strain = (deriv_shift + deriv_shift.T) / 2.0
    return np.einsum(
        "i,ij", face_normal, longitudinal_operator_flat_cartesian(strain)
    )
