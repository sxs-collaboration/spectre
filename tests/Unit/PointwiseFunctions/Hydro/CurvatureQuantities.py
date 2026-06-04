# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def ricci_in_gr(stress_energy, spacetime_metric):
    # T_{ab} = g_{ac} g_{bd} T^{cd}
    stress_energy_lower = np.einsum(
        "ac,bd,cd->ab", spacetime_metric, spacetime_metric, stress_energy
    )
    # T = g_{ab} T^{ab}
    trace = np.einsum("ab,ab->", spacetime_metric, stress_energy)
    # R_{ab} = 8pi (T_{ab} - 1/2 g_{ab} T)
    return 8.0 * np.pi * (stress_energy_lower - 0.5 * spacetime_metric * trace)


def ricci_scalar(ricci_tensor, inverse_spacetime_metric):
    # R = g^{ab} R_{ab}
    return np.einsum("ab,ab->", inverse_spacetime_metric, ricci_tensor)
