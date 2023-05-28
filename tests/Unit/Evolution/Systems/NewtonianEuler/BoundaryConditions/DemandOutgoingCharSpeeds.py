# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def error(
    face_mesh_velocity,
    normal_covector,
    int_mass_density,
    int_velocity,
    int_specific_internal_energy,
    use_polytropic_eos,
):
    if use_polytropic_eos:
        polytropic_constant = 1.4
        polytropic_exponent = 5.0 / 3.0
        sound_speed = np.sqrt(
            polytropic_constant
            * polytropic_exponent
            * pow(int_mass_density, polytropic_exponent - 1.0)
        )
    else:
        adiabatic_index = 1.3
        chi = int_specific_internal_energy * (adiabatic_index - 1.0)
        kappa_times_p_over_rho_squared = (
            adiabatic_index - 1.0
        ) ** 2 * int_specific_internal_energy
        sound_speed = np.sqrt(chi + kappa_times_p_over_rho_squared)

    normal_dot_velocity = np.einsum("i,i", int_velocity, normal_covector)

    if face_mesh_velocity is None:
        min_char_speed = normal_dot_velocity - sound_speed
    else:
        normal_dot_mesh_velocity = np.einsum(
            "i,i", face_mesh_velocity, normal_covector
        )

        min_char_speed = (
            normal_dot_velocity - normal_dot_mesh_velocity - sound_speed
        )

    if min_char_speed < 0.0:
        return "DemandOutgoingCharSpeeds boundary condition violated"

    return None
