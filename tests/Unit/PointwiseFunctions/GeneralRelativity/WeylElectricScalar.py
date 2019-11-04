# Distributed under the MIT License.
# See LICENSE.txt for details.
import numpy as np


def weyl_electric_scalar(weyl_electric, inverse_spatial_metric):
    return (np.einsum("ik,jl,ij,kl", weyl_electric, weyl_electric,
                      inverse_spatial_metric, inverse_spatial_metric))
