# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def transverse_projection_operator(spatial_metric_or_its_inverse,
                                   normal_vector_or_one_form):
    dim_offset = np.shape(normal_vector_or_one_form)[0] - np.shape(
        spatial_metric_or_its_inverse)[1]
    if dim_offset != 0 and dim_offset != 1:
        raise RuntimeError("Incompatible inputs passed")
    return (spatial_metric_or_its_inverse -
            np.einsum('i,j->ij', normal_vector_or_one_form[dim_offset:],
                      normal_vector_or_one_form[dim_offset:]))


def transverse_projection_operator_mixed_from_spatial_input(
    normal_vector, normal_one_form):
    return (np.eye(np.shape(normal_vector)[0]) -
            np.einsum('i,j->ij', normal_vector, normal_one_form))
