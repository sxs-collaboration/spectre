# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def dt_psi0(
    psi_monopole,
    psi_dipole,
    psi_quadrupole,
    dt_psi_dipole,
    inverse_spacetime_metric,
    trace_christoffel,
    evolved_vars,
    wt_radius,
):
    return evolved_vars[1]


def dt2_psi0(
    psi_monopole,
    psi_dipole,
    psi_quadrupole,
    dt_psi_dipole,
    inverse_spacetime_metric,
    trace_christoffel,
    evolved_vars,
    wt_radius,
):
    psi0 = evolved_vars[0]
    dt_psi0 = evolved_vars[1]
    result = -2 * np.dot(inverse_spacetime_metric[0, 1:], dt_psi_dipole)
    result -= 2.0 * np.einsum(
        "ij,ij", inverse_spacetime_metric[1:, 1:], psi_quadrupole
    )
    result -= (
        2
        * np.trace(inverse_spacetime_metric[1:, 1:])
        * (psi_monopole - psi0)
        / wt_radius**2
    )
    result += trace_christoffel[0] * dt_psi0
    result += np.dot(trace_christoffel[1:], psi_dipole)
    return result / inverse_spacetime_metric[0][0]
