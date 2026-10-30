# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from Elasticity.ConstitutiveRelations.IsotropicHomogeneous import stress


def potential_energy_density(strain, coordinates, bulk_modulus, shear_modulus):
    local_stress = stress(strain, coordinates, bulk_modulus, shear_modulus)
    return -0.5 * np.einsum("ij, ij", strain, local_stress)


# The bulk and shear parts below are computed with the volumetric and
# deviatoric projectors instead of the C++ route (traces multiplied directly
# for the bulk part; explicit deviators summed over the upper triangle with a
# factor 2 off the diagonal for the shear part). The traces run over the Dim
# dimensions of the input, as in the C++.


def bulk_potential_energy_density(strain, stress):
    # -1/2 S_vol : T_vol with A_vol = (tr A / Dim) delta
    dim = strain.shape[0]
    krond = np.eye(dim)
    strain_vol = np.trace(strain) / dim * krond
    stress_vol = np.trace(stress) / dim * krond
    return -0.5 * np.einsum("ij,ij", strain_vol, stress_vol)


def deviatoric_projector(dim):
    # P_ijkl = (delta_ik delta_jl + delta_il delta_jk) / 2
    #          - delta_ij delta_kl / Dim
    krond = np.eye(dim)
    return (
        0.5
        * (
            np.einsum("ik,jl->ijkl", krond, krond)
            + np.einsum("il,jk->ijkl", krond, krond)
        )
        - np.einsum("ij,kl->ijkl", krond, krond) / dim
    )


def shear_potential_energy_density(strain, stress):
    # -1/2 S_ij P_ijkl T_kl. No deviator is formed; the factor 2 of the
    # off-diagonal terms comes from the ij and ji terms of the full contraction
    return -0.5 * np.einsum(
        "ij,ijkl,kl", strain, deviatoric_projector(strain.shape[0]), stress
    )
