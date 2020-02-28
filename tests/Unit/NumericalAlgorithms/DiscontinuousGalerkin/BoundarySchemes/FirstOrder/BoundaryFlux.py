# Distributed under the MIT License.
# See LICENSE.txt for details.


def boundary_flux(field_int, field_ext):
    # We are working with a simple "central" numerical flux
    numerical_flux = 0.5 * (field_int + field_ext)
    # Eq. 2.20 in https://arxiv.org/pdf/1510.01190.pdf
    surface_term = numerical_flux - field_int
    # Eq. 3.19 in https://arxiv.org/pdf/1510.01190.pdf
    # (The LGL quadrature weight for 5 points at the boundary is 1/10)
    return -surface_term * 10.
