# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def weyl_magnetic_tensor(grad_extrinsic_curvature, spatial_metric):
    e_ijk = np.zeros((3, 3, 3))
    e_ijk[0, 1, 2] = e_ijk[1, 2, 0] = e_ijk[2, 0, 1] = 1.0
    e_ijk[0, 2, 1] = e_ijk[2, 1, 0] = e_ijk[1, 0, 2] = -1.0

    gamma = np.linalg.det(spatial_metric)
    result = ( 0.5 / np.sqrt(gamma) ) * \
        (np.einsum("kli,mlk,jm", grad_extrinsic_curvature, \
                   e_ijk, spatial_metric) + \
             np.einsum("klj,mlk,im", grad_extrinsic_curvature, \
                       e_ijk, spatial_metric))
    return result
