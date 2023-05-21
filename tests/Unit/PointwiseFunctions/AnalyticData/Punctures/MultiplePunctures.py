# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

masses = [1.0, 0.5]
centers = np.array([[1.0, 2.0, 3.0], [-1.0, 2.0, 3.0]])
dimensionless_momenta = np.array([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]])
dimensionless_spins = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.1]])


def _traceless_conformal_extrinsic_curvature(r, n, P, S):
    n_dot_P = np.dot(n, P)
    S_cross_n = np.cross(S, n)
    A = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            A[i, j] = (
                3.0
                / 2.0
                / r**2
                * (
                    P[i] * n[j]
                    + P[j] * n[i]
                    - (np.eye(3)[i, j] - n[i] * n[j]) * n_dot_P
                    + 2.0 / r * (n[i] * S_cross_n[j] + n[j] * S_cross_n[i])
                )
            )
    return A


def traceless_conformal_extrinsic_curvature(x):
    A_sum = np.zeros((3, 3))
    for i in range(len(masses)):
        r = np.linalg.norm(x - centers[i])
        n = (x - centers[i]) / r
        A_sum += _traceless_conformal_extrinsic_curvature(
            r=r,
            n=n,
            P=masses[i] * dimensionless_momenta[i],
            S=masses[i] ** 2 * dimensionless_spins[i],
        )
    return A_sum


def alpha(x):
    one_over_alpha = 0.0
    for i in range(len(masses)):
        r = np.linalg.norm(x - centers[i])
        one_over_alpha += 0.5 * masses[i] / r
    return 1.0 / one_over_alpha


def beta(x):
    A = traceless_conformal_extrinsic_curvature(x)
    return 1.0 / 8.0 * alpha(x) ** 7 * np.einsum("ij,ij", A, A)
