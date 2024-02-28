# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def delta(r_sqrd, m, a):
    return r_sqrd - 2.0 * m * np.sqrt(r_sqrd) + a**2


def transformation_matrix(x, a):
    # Coordinate transformation matrix from SKS to KS.
    # Returns the P^j_ibar from equation 10 of Spherical-KS documentation.
    # Note, this assuemes the black hole spin is along z-direction.
    r_sqrd = boyer_lindquist_r_sqrd(x)
    r = np.sqrt(r_sqrd)
    rho = np.sqrt(r_sqrd + a**2)
    P_ji = np.diag([rho / r, rho / r, 1.0])
    return P_ji


def jacobian_matrix(x, a):
    # Jacbian matrix for coordinate transformation from SKS to KS.
    # Returns T^i_jbar from equation 16 of Spherical-KS documentation.
    # Note, this assuemes the black hole spin is along z-direction.
    P_ij = transformation_matrix(x, a)

    r_sqrd = boyer_lindquist_r_sqrd(x)
    r = np.sqrt(r_sqrd)
    rho = np.sqrt(r_sqrd + a**2)

    F_ik = (-1.0 / (rho * r**3)) * np.diag([a**2, a**2, 0.0])
    x_vec = np.array(x).T

    T_ij = P_ij + np.outer(np.matmul(F_ik, x_vec), x_vec.T)

    return T_ij.T


def inverse_jacobian_matrix(x, a):
    # Inverse Jacbian matrix for coordinate transformation
    # from KS to Spherical KS.
    # Returns S^i_jbar from equation 17 of Spherical-KS documentation.
    # Note, this assuemes the black hole spin is along z-direction.
    T_ij = jacobian_matrix(x, a)
    S_ij = np.linalg.inv(T_ij)
    return S_ij


def sigma(r_sqrd, sin_theta_sqrd, a):
    return r_sqrd + (1.0 - sin_theta_sqrd) * a**2


def ucase_a(r_sqrd, sin_theta_sqrd, m, a):
    return (r_sqrd + a**2) ** 2 - delta(
        r_sqrd, m, a
    ) * sin_theta_sqrd * a**2


def boyer_lindquist_gtf(r_sqrd, sin_theta_sqrd, m, a):
    return (
        -2.0
        * m
        * np.sqrt(r_sqrd)
        * a
        * sin_theta_sqrd
        / sigma(r_sqrd, sin_theta_sqrd, a)
    )


def boyer_lindquist_gff(r_sqrd, sin_theta_sqrd, m, a):
    return (
        ucase_a(r_sqrd, sin_theta_sqrd, m, a)
        * sin_theta_sqrd
        / sigma(r_sqrd, sin_theta_sqrd, a)
    )


def boyer_lindquist_r_sqrd(x):
    # Here, we assume x is in Spherical-KS coordinates
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2


def boyer_lindquist_sin_theta_sqrd(z_sqrd, r_sqrd):
    return 1.0 - z_sqrd / r_sqrd


def kerr_schild_h(x, m, a):
    r_sqrd = boyer_lindquist_r_sqrd(x)
    return m * r_sqrd * np.sqrt(r_sqrd) / (r_sqrd**2 + a**2 * x[2] ** 2)


def kerr_schild_spatial_null_vec(x, m, a):
    r_sqrd = boyer_lindquist_r_sqrd(x)
    r = np.sqrt(r_sqrd)
    rho = np.sqrt(r_sqrd + a**2)
    denom = 1.0 / rho**2
    # Again, we assume Spherical-KS coordinates
    # xbar/r = x/rho, ybar/r = y/rho
    # where rho^2 = r^2+a^2 and xbar and ybar are Spherical-KS
    # Thus, we need to stick in the converseion factor of rho/r.
    conv_fac = rho / r
    l_vec = np.array(
        [
            (r * x[0] * conv_fac + a * x[1] * conv_fac) * denom,
            (r * x[1] * conv_fac - a * x[0] * conv_fac) * denom,
            x[2] / r,
        ]
    )
    return l_vec


def sph_kerr_schild_spatial_null_form(x, m, a):
    jac = jacobian_matrix(x, a)
    l_vec = kerr_schild_spatial_null_vec(x, m, a)
    return jac @ l_vec


def sph_kerr_schild_spatial_null_vec(x, m, a):
    inv_jac = inverse_jacobian_matrix(x, a)
    l_vec = kerr_schild_spatial_null_vec(x, m, a)
    return inv_jac.T @ l_vec


def kerr_schild_lapse(x, m, a):
    null_vector_0 = -1.0
    return np.sqrt(
        1.0
        / (1.0 + 2.0 * kerr_schild_h(x, m, a) * null_vector_0 * null_vector_0)
    )


def sph_kerr_schild_shift(x, m, a):
    null_vector_0 = -1.0
    return (
        -2.0
        * kerr_schild_h(x, m, a)
        * null_vector_0
        * kerr_schild_lapse(x, m, a) ** 2
    ) * sph_kerr_schild_spatial_null_vec(x, m, a)


def sph_kerr_schild_spatial_metric(x, m, a):
    prefactor = 2.0 * kerr_schild_h(x, m, a)
    null_form = sph_kerr_schild_spatial_null_form(x, m, a)
    T_ij = jacobian_matrix(x, a)
    result = T_ij @ T_ij.T + prefactor * np.outer(null_form, null_form)
    return result


def angular_momentum(m, a, rmax):
    return np.sqrt(m) * (
        (np.power(rmax, 1.5) + a * np.sqrt(m))
        * (a**2 - 2.0 * a * np.sqrt(m) * np.sqrt(rmax) + rmax**2)
        / (
            2.0 * a * np.sqrt(m) * np.power(rmax, 1.5)
            + (rmax - 3.0 * m) * rmax**2
        )
    )


def angular_velocity(angular_momentum, r_sqrd, sin_theta_sqrd, m, a):
    prefactor = (
        2.0
        * angular_momentum
        * delta(r_sqrd, m, a)
        * sin_theta_sqrd
        / np.power(boyer_lindquist_gff(r_sqrd, sin_theta_sqrd, m, a), 2.0)
    )
    return prefactor / (
        1.0 + np.sqrt(1.0 + 2.0 * angular_momentum * prefactor)
    ) - (
        boyer_lindquist_gtf(r_sqrd, sin_theta_sqrd, m, a)
        / boyer_lindquist_gff(r_sqrd, sin_theta_sqrd, m, a)
    )


def u_t(angular_momentum, r_sqrd, sin_theta_sqrd, m, a):
    return np.sqrt(
        angular_momentum
        / (
            boyer_lindquist_gtf(r_sqrd, sin_theta_sqrd, m, a)
            + angular_velocity(angular_momentum, r_sqrd, sin_theta_sqrd, m, a)
            * boyer_lindquist_gff(r_sqrd, sin_theta_sqrd, m, a)
        )
    )


def potential(angular_momentum, r_sqrd, sin_theta_sqrd, m, a):
    return angular_momentum * angular_velocity(
        angular_momentum, r_sqrd, sin_theta_sqrd, m, a
    ) - np.log(u_t(angular_momentum, r_sqrd, sin_theta_sqrd, m, a))


def specific_enthalpy(
    x,
    t,
    bh_mass,
    bh_dimless_a,
    dimless_r_in,
    dimless_r_max,
    polytropic_constant,
    polytropic_exponent,
):
    r_in = bh_mass * dimless_r_in
    bh_spin_a = bh_mass * bh_dimless_a
    l = angular_momentum(bh_mass, bh_spin_a, bh_mass * dimless_r_max)
    Win = potential(l, r_in * r_in, 1.0, bh_mass, bh_spin_a)
    r_sqrd = boyer_lindquist_r_sqrd(x)
    sin_theta_sqrd = boyer_lindquist_sin_theta_sqrd(x[2] * x[2], r_sqrd)
    result = 1.0
    if np.sqrt(r_sqrd * sin_theta_sqrd) >= r_in:
        W = potential(l, r_sqrd, sin_theta_sqrd, bh_mass, bh_spin_a)
        if W < Win:
            result = np.exp(Win - W)

    return result


def rest_mass_density(
    x,
    t,
    bh_mass,
    bh_dimless_a,
    dimless_r_in,
    dimless_r_max,
    polytropic_constant,
    polytropic_exponent,
):
    return np.power(
        (polytropic_exponent - 1.0)
        * (
            specific_enthalpy(
                x,
                t,
                bh_mass,
                bh_dimless_a,
                dimless_r_in,
                dimless_r_max,
                polytropic_constant,
                polytropic_exponent,
            )
            - 1.0
        )
        / (polytropic_exponent * polytropic_constant),
        1.0 / (polytropic_exponent - 1.0),
    )


def specific_internal_energy(
    x,
    t,
    bh_mass,
    bh_dimless_a,
    dimless_r_in,
    dimless_r_max,
    polytropic_constant,
    polytropic_exponent,
):
    return (
        polytropic_constant
        * np.power(
            rest_mass_density(
                x,
                t,
                bh_mass,
                bh_dimless_a,
                dimless_r_in,
                dimless_r_max,
                polytropic_constant,
                polytropic_exponent,
            ),
            polytropic_exponent - 1.0,
        )
        / (polytropic_exponent - 1.0)
    )


def pressure(
    x,
    t,
    bh_mass,
    bh_dimless_a,
    dimless_r_in,
    dimless_r_max,
    polytropic_constant,
    polytropic_exponent,
):
    return polytropic_constant * np.power(
        rest_mass_density(
            x,
            t,
            bh_mass,
            bh_dimless_a,
            dimless_r_in,
            dimless_r_max,
            polytropic_constant,
            polytropic_exponent,
        ),
        polytropic_exponent,
    )


def spatial_velocity(
    x,
    t,
    bh_mass,
    bh_dimless_a,
    dimless_r_in,
    dimless_r_max,
    polytropic_constant,
    polytropic_exponent,
):
    r_in = bh_mass * dimless_r_in
    bh_spin_a = bh_mass * bh_dimless_a
    l = angular_momentum(bh_mass, bh_spin_a, bh_mass * dimless_r_max)
    Win = potential(l, r_in * r_in, 1.0, bh_mass, bh_spin_a)
    r_sqrd = boyer_lindquist_r_sqrd(x)
    sin_theta_sqrd = boyer_lindquist_sin_theta_sqrd(x[2] * x[2], r_sqrd)
    sks_to_ks_factor = np.sqrt(r_sqrd + bh_spin_a**2) / np.sqrt(r_sqrd)

    result = np.array([0.0, 0.0, 0.0])
    if np.sqrt(r_sqrd * sin_theta_sqrd) >= r_in:
        W = potential(l, r_sqrd, sin_theta_sqrd, bh_mass, bh_spin_a)
        if W < Win:
            transport_velocity_ks = (
                sks_to_ks_factor
                * np.array([-x[1], x[0], 0.0])
                * angular_velocity(
                    l, r_sqrd, sin_theta_sqrd, bh_mass, bh_spin_a
                )
            )
            transport_velocity_sks = (
                inverse_jacobian_matrix(x, bh_spin_a) @ transport_velocity_ks
            )
            result += (
                transport_velocity_sks
                + sph_kerr_schild_shift(x, bh_mass, bh_spin_a)
            ) / kerr_schild_lapse(x, bh_mass, bh_spin_a)
    return np.array(result)


def lorentz_factor(
    x,
    t,
    bh_mass,
    bh_dimless_a,
    dimless_r_in,
    dimless_r_max,
    polytropic_constant,
    polytropic_exponent,
):
    bh_spin_a = bh_mass * bh_dimless_a
    spatial_metric = sph_kerr_schild_spatial_metric(x, bh_mass, bh_spin_a)
    velocity = spatial_velocity(
        x,
        t,
        bh_mass,
        bh_dimless_a,
        dimless_r_in,
        dimless_r_max,
        polytropic_constant,
        polytropic_exponent,
    )
    return 1.0 / np.sqrt(
        1.0 - np.einsum("i,j,ij", velocity, velocity, spatial_metric)
    )


def magnetic_field(*args):
    return np.array([0.0, 0.0, 0.0])


def divergence_cleaning_field(*args):
    return 0.0
