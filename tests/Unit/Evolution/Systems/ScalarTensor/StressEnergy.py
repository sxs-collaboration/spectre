# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def trace_reversed_stress_energy(pi_scalar, phi_scalar, lapse):
    # Define the different components
    tt_component = lapse * lapse * pi_scalar * pi_scalar
    tj_component = -lapse * pi_scalar * phi_scalar
    ij_component = np.outer(phi_scalar, phi_scalar)

    # Construct the trace-reversed stress energy tensor
    stress_energy = np.zeros((4, 4))

    stress_energy[0, 0] = tt_component
    stress_energy[0, 1:] = tj_component
    stress_energy[1:, 0] = stress_energy[0, 1:]
    stress_energy[1:, 1:] = ij_component

    return stress_energy


def add_stress_energy_term_to_dt_pi(trace_reversed_stress_energy, lapse):
    return 0.1234 - 16.0 * np.pi * lapse * trace_reversed_stress_energy
