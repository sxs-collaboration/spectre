# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def interface_outgoing_null_normal(spacetime_normal_vector_or_one_form,
                                   interface_normal_vector_or_one_form):
    result = (2.**-0.5) * spacetime_normal_vector_or_one_form
    result[1:] = result[1:] + (2.**-0.5) * interface_normal_vector_or_one_form
    return result


def interface_incoming_null_normal(spacetime_normal_vector_or_one_form,
                                   interface_normal_vector_or_one_form):
    result = (2.**-0.5) * spacetime_normal_vector_or_one_form
    result[1:] = result[1:] - (2.**-0.5) * interface_normal_vector_or_one_form
    return result
