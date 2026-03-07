# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def interface_outgoing_null_normal(
    spacetime_normal_vector_or_one_form,
    interface_normal_vector_or_one_form,
    shift=None,
):
    result = (2.0**-0.5) * spacetime_normal_vector_or_one_form
    if shift is not None:
        result[0] = result[0] + (2.0**-0.5) * np.einsum(
            "i...,i...->...",
            interface_normal_vector_or_one_form,
            shift,
        )
    result[1:] = (
        result[1:] + (2.0**-0.5) * interface_normal_vector_or_one_form
    )
    return result


def interface_incoming_null_normal(
    spacetime_normal_vector_or_one_form,
    interface_normal_vector_or_one_form,
    shift=None,
):
    result = (2.0**-0.5) * spacetime_normal_vector_or_one_form
    if shift is not None:
        result[0] = result[0] - (2.0**-0.5) * np.einsum(
            "i...,i...->...",
            interface_normal_vector_or_one_form,
            shift,
        )
    result[1:] = (
        result[1:] - (2.0**-0.5) * interface_normal_vector_or_one_form
    )
    return result
