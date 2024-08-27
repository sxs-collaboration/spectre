# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def trace_reversed_stress_energy(pi_scalar, phi_scalar, lapse, shift):
    # Define the different components
    rho = pi_scalar * pi_scalar
    j_vec = -pi_scalar * phi_scalar
    S = np.outer(phi_scalar, phi_scalar)

    # Construct the trace-reversed stress energy tensor
    stress_energy = np.zeros((4, 4))

    # 00-component
    stress_energy[0, 0] = (
        np.power(lapse, 2) * rho
        + 2.0 * lapse * np.dot(shift, j_vec)
        + shift @ S @ shift
    )

    # 0i-component
    stress_energy[0, 1:] = lapse * j_vec + shift @ S
    stress_energy[1:, 0] = np.transpose(stress_energy[0, 1:])

    # ij-component
    stress_energy[1:, 1:] = S

    return stress_energy


def add_stress_energy_term_to_dt_pi(trace_reversed_stress_energy, lapse):
    return 0.1234 - 16.0 * np.pi * lapse * trace_reversed_stress_energy
