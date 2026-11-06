# Distributed under the MIT License.
# See LICENSE.txt for details.

import itertools

import numpy as np

# Voigt index of each pair of Cartesian indices: [xx, yy, zz, yz, xz, xy]
VOIGT_INDEX = {
    (0, 0): 0,
    (1, 1): 1,
    (2, 2): 2,
    (1, 2): 3,
    (2, 1): 3,
    (0, 2): 4,
    (2, 0): 4,
    (0, 1): 5,
    (1, 0): 5,
}

# Symmetric positive definite Voigt matrix with distinct shear moduli and
# nonzero normal-shear and shear-shear couplings. Must match `c_general()` in
# Test_AnisotropicHomogeneous.cpp.
C_GENERAL = np.array(
    [
        [200.0, 60.0, 50.0, 10.0, -5.0, 8.0],
        [60.0, 180.0, 55.0, -7.0, 6.0, -4.0],
        [50.0, 55.0, 160.0, 5.0, 9.0, 3.0],
        [10.0, -7.0, 5.0, 70.0, -3.0, 2.0],
        [-5.0, 6.0, 9.0, -3.0, 60.0, -4.0],
        [8.0, -4.0, 3.0, 2.0, -4.0, 50.0],
    ]
)


def stiffness_tensor(voigt_matrix):
    # Full fourth-rank tensor C_ijkl = C_{V(ij) V(kl)}. With this map the
    # factor 2 of the engineering shear strain arises from the two terms
    # S_kl and S_lk of the contraction, so no Voigt strain vector is formed.
    result = np.zeros((3, 3, 3, 3))
    for i, j, k, l in itertools.product(range(3), repeat=4):
        result[i, j, k, l] = voigt_matrix[VOIGT_INDEX[i, j], VOIGT_INDEX[k, l]]
    return result


def stress_general(strain, x):
    return -np.einsum("ijkl,kl->ij", stiffness_tensor(C_GENERAL), strain)
