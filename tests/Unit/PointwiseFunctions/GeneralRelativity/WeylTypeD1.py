# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def weyl_type_D1(weyl_electric, spatial_metric, inverse_spatial_metric):
    inverse_weyl_electric = np.einsum(
        "lk,ik,lj",
        weyl_electric,
        inverse_spatial_metric,
        inverse_spatial_metric,
    )
    a = 16 * (np.einsum("ij,ij", weyl_electric, inverse_weyl_electric))
    b = -64 * (
        np.einsum(
            "il,lk,ij,jk",
            weyl_electric,
            inverse_spatial_metric,
            inverse_weyl_electric,
            weyl_electric,
        )
    )
    return (
        (a / 12) * (np.einsum("ij", spatial_metric))
        - (np.einsum("ij", weyl_electric) * (b / a))
        - 4
        * np.einsum(
            "im,km,jk", weyl_electric, inverse_spatial_metric, weyl_electric
        )
    )
