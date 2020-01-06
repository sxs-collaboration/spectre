# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

# Functions for testing PenaltyFlux.hpp
penalty_factor = 1.0


def pi_penalty_flux(n_dot_flux_pi_int, n_dot_flux_phi_int, v_plus_int,
                    v_minus_int, unit_normal_int, minus_n_dot_flux_pi_ext,
                    minus_n_dot_flux_phi_ext, v_plus_ext, v_minus_ext,
                    unit_normal_ext):
    return n_dot_flux_pi_int + 0.5 * penalty_factor * (v_minus_int -
                                                       v_plus_ext)


def phi_penalty_flux(n_dot_flux_pi_int, n_dot_flux_phi_int, v_plus_int,
                     v_minus_int, unit_normal_int, minus_n_dot_flux_pi_ext,
                     minus_n_dot_flux_phi_ext, v_plus_ext, v_minus_ext,
                     unit_normal_ext):
    return n_dot_flux_phi_int - 0.5 * penalty_factor * (\
        unit_normal_int * v_minus_int + unit_normal_ext * v_plus_ext)


def psi_penalty_flux(n_dot_flux_pi_int, n_dot_flux_phi_int, v_plus_int,
                     v_minus_int, unit_normal_int, minus_n_dot_flux_pi_ext,
                     minus_n_dot_flux_phi_ext, v_plus_ext, v_minus_ext,
                     unit_normal_ext):
    return n_dot_flux_pi_int * 0


# End functions for testing PenaltyFlux.hpp
