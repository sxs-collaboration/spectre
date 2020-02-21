# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


# Test functions for characteristic speeds
def char_speed_vpsi(unit_normal):
    return 0.


def char_speed_vzero(unit_normal):
    return 0.


def char_speed_vplus(unit_normal):
    return 1.


def char_speed_vminus(unit_normal):
    return -1.


# End test functions for characteristic speeds


# Test functions for characteristic fields
def char_field_vpsi(gamma2, psi, pi, phi, normal_one_form):
    return psi


def char_field_vzero(gamma2, psi, pi, phi, normal_one_form):
    normal_vector = normal_one_form
    projection_tensor = np.identity(len(normal_vector)) -\
        np.einsum('i,j', normal_one_form, normal_vector)
    return np.einsum('ij,j->i', projection_tensor, phi)


def char_field_vplus(gamma2, psi, pi, phi, normal_one_form):
    normal_vector = normal_one_form
    phi_dot_normal = np.einsum('i,i->', normal_vector, phi)
    return pi + phi_dot_normal - (gamma2 * psi)


def char_field_vminus(gamma2, psi, pi, phi, normal_one_form):
    normal_vector = normal_one_form
    phi_dot_normal = np.einsum('i,i->', normal_vector, phi)
    return pi - phi_dot_normal - (gamma2 * psi)


# End test functions for characteristic fields


# Test functions for evolved fields
def evol_field_psi(gamma2, vpsi, vzero, vplus, vminus, normal_one_form):
    return vpsi


def evol_field_pi(gamma2, vpsi, vzero, vplus, vminus, normal_one_form):
    return 0.5 * (vplus + vminus) + gamma2 * vpsi


def evol_field_phi(gamma2, vpsi, vzero, vplus, vminus, normal_one_form):
    return 0.5 * (vplus - vminus) * normal_one_form + vzero


# End test functions for evolved fields
