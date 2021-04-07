# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def to_different_frame(src, jacobian):
    # Jacobian here is d^x_src/d^x_dest
    return np.einsum("ab,ac,bd", src, jacobian, jacobian)


def first_index_to_different_frame(src, jacobian):
    # Jacobian here is d^x_src/d^x_dest
    return np.einsum("abc,ad->dbc", src, jacobian)
