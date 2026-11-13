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


def weyl_electric(
    vacuum_weyl_electric,
    _stress_energy,
    ricci_tensor,
    ricci_scalar,
    inverse_spacetime_metric,
    induced_spatial_metric,
):
    # Mixed-index four-spatial metric: gamma_a^B = g^{BC} gamma_{aC}
    gamma_mixed = np.einsum(
        "BC,aC->aB", inverse_spacetime_metric, induced_spatial_metric
    )
    # Spatial projection of Ricci: gamma_a^c gamma_b^d R_{cd}
    proj_R = np.einsum("ac,bd,cd->ab", gamma_mixed, gamma_mixed, ricci_tensor)
    # Raised four-spatial metric: gamma^{cd} = g^{ca} g^{db} gamma_{ab}
    gamma_upper = np.einsum(
        "ca,db,ab->cd",
        inverse_spacetime_metric,
        inverse_spacetime_metric,
        induced_spatial_metric,
    )
    # Trace: gamma^{cd} R_{cd}
    trace_gamma_R = np.einsum("cd,cd->", gamma_upper, ricci_tensor)
    # Spacetime matter contribution to Weyl electric (4x4)
    matter_weyl = (
        -0.5 * (proj_R + induced_spatial_metric * trace_gamma_R)
        + induced_spatial_metric * ricci_scalar / 3.0
    )
    # Extract spatial components and add vacuum contribution
    return vacuum_weyl_electric + matter_weyl[1:, 1:]


def kretschmann_scalar(
    weyl_electric_scalar,
    weyl_magnetic_scalar,
    inverse_spacetime_metric,
    ricci_tensor,
    ricci_scalar,
):
    # K = 8(E - B) + 2 R_{ab} R^{ab} - R^2/3
    kretschmann_vacuum = 8.0 * (weyl_electric_scalar - weyl_magnetic_scalar)
    ricci_sq = np.einsum(
        "ab,ac,bd,cd->",
        ricci_tensor,
        inverse_spacetime_metric,
        inverse_spacetime_metric,
        ricci_tensor,
    )
    return kretschmann_vacuum + 2.0 * ricci_sq - ricci_scalar**2 / 3.0
