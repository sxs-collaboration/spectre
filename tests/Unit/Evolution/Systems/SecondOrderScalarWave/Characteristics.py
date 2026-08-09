# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Test functions for characteristic speeds
def char_speed_vzero(unit_normal):
    return 0.0


def char_speed_vplus(unit_normal):
    return 1.0


def char_speed_vminus(unit_normal):
    return -1.0


# Test functions for characteristic fields
def char_field_vzero(pi, phi, normal_one_form):
    normal_vector = normal_one_form
    projection_tensor = np.identity(len(normal_vector)) - np.einsum(
        "i,j", normal_one_form, normal_vector
    )
    return np.einsum("ij,j->i", projection_tensor, phi)


def char_field_vplus(pi, phi, normal_one_form):
    normal_vector = normal_one_form
    phi_dot_normal = np.einsum("i,i->", normal_vector, phi)
    return pi + phi_dot_normal


def char_field_vminus(pi, phi, normal_one_form):
    normal_vector = normal_one_form
    phi_dot_normal = np.einsum("i,i->", normal_vector, phi)
    return pi - phi_dot_normal


# End test functions for characteristic fields


# Test functions for the inverse characteristic transform
def inverse_field_pi(vzero, vplus, vminus, normal_one_form):
    return 0.5 * (vplus + vminus)


def inverse_field_phi(vzero, vplus, vminus, normal_one_form):
    return 0.5 * (vplus - vminus) * normal_one_form + vzero


# End test functions for the inverse characteristic transform
