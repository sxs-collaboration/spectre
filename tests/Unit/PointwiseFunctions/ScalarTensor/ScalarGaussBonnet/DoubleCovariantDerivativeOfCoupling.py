# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def DDCoupling_normal_normal_projection(
    coupling_prime, coupling_prime_prime, Pi, nnDDPsi
):
    return coupling_prime_prime * Pi**2 + coupling_prime * (nnDDPsi)


def DDCoupling_normal_spatial_projection(
    coupling_prime, coupling_prime_prime, Pi, DPsi, nsDDPsi
):
    return -coupling_prime_prime * Pi * DPsi + coupling_prime * nsDDPsi


def DDCoupling_spatial_spatial_projection(
    coupling_prime, coupling_prime_prime, DPsi, ssDDPsi
):
    return (
        coupling_prime_prime * np.tensordot(DPsi, DPsi, axes=0)
        + coupling_prime * ssDDPsi
    )


def DDCoupling_spatial_trace(inverse_spatial_metric, ssDDFPsi):
    return np.einsum("ij,ji->", inverse_spatial_metric, ssDDFPsi)
