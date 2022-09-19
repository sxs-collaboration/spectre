# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def to_different_frame(src, jacobian):
    # Jacobian here is d^x_src/d^x_dest
    return np.einsum("ab,ac,bd", src, jacobian, jacobian)


def to_different_frame_Scalar(src, jacobian):
    # Jacobian here is d^x_src/d^x_dest
    return src


def to_different_frame_I(src, jacobian):
    # Jacobian here is d^x_src/d^x_dest
    inv_jacobian = np.linalg.inv(jacobian)
    return np.einsum("a,ca", src, inv_jacobian)


def to_different_frame_i(src, jacobian):
    # Jacobian here is d^x_src/d^x_dest
    return np.einsum("a,ac", src, jacobian)


def to_different_frame_iJ(src, jacobian):
    # Jacobian here is d^x_src/d^x_dest
    inv_jacobian = np.linalg.inv(jacobian)
    return np.einsum("ab,ac,db", src, jacobian, inv_jacobian)


def to_different_frame_ii(src, jacobian):
    # Jacobian here is d^x_src/d^x_dest
    return np.einsum("ab,ac,bd", src, jacobian, jacobian)


def to_different_frame_II(src, jacobian):
    # Jacobian here is d^x_src/d^x_dest
    inv_jacobian = np.linalg.inv(jacobian)
    return np.einsum("ab,ca,db", src, inv_jacobian, inv_jacobian)


def to_different_frame_ijj(src, jacobian):
    # Jacobian here is d^x_src/d^x_dest
    return np.einsum("abe,ac,bd,ef", src, jacobian, jacobian, jacobian)


def first_index_to_different_frame(src, jacobian):
    # Jacobian here is d^x_src/d^x_dest
    return np.einsum("abc,ad->dbc", src, jacobian)
