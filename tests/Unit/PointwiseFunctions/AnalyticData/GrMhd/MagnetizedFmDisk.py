# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import PointwiseFunctions.AnalyticSolutions.RelativisticEuler.FishboneMoncriefDisk as fm_disk
import PointwiseFunctions.GeneralRelativity.KerrSchildCoords as ks_coords


def rest_mass_density(
    x,
    bh_mass,
    bh_dimless_spin,
    dimless_r_in,
    dimless_r_max,
    polytropic_constant,
    polytropic_exponent,
    noise,
    threshold_density,
    plasma_beta,
):
    dummy_time = 0.0
    return fm_disk.rest_mass_density(
        x,
        dummy_time,
        bh_mass,
        bh_dimless_spin,
        dimless_r_in,
        dimless_r_max,
        polytropic_constant,
        polytropic_exponent,
        noise,
    )


def spatial_velocity(
    x,
    bh_mass,
    bh_dimless_spin,
    dimless_r_in,
    dimless_r_max,
    polytropic_constant,
    polytropic_exponent,
    noise,
    threshold_density,
    plasma_beta,
):
    dummy_time = 0.0
    return fm_disk.spatial_velocity(
        x,
        dummy_time,
        bh_mass,
        bh_dimless_spin,
        dimless_r_in,
        dimless_r_max,
        polytropic_constant,
        polytropic_exponent,
        noise,
    )


def specific_internal_energy(
    x,
    bh_mass,
    bh_dimless_spin,
    dimless_r_in,
    dimless_r_max,
    polytropic_constant,
    polytropic_exponent,
    noise,
    threshold_density,
    plasma_beta,
):
    dummy_time = 0.0
    return fm_disk.specific_internal_energy(
        x,
        dummy_time,
        bh_mass,
        bh_dimless_spin,
        dimless_r_in,
        dimless_r_max,
        polytropic_constant,
        polytropic_exponent,
        noise,
    )


def pressure(
    x,
    bh_mass,
    bh_dimless_spin,
    dimless_r_in,
    dimless_r_max,
    polytropic_constant,
    polytropic_exponent,
    noise,
    threshold_density,
    plasma_beta,
):
    dummy_time = 0.0
    return fm_disk.pressure(
        x,
        dummy_time,
        bh_mass,
        bh_dimless_spin,
        dimless_r_in,
        dimless_r_max,
        polytropic_constant,
        polytropic_exponent,
        noise,
    )


def magnetic_potential(
    r,
    sin_theta_sqrd,
    m,
    a,
    rin,
    rmax,
    polytropic_constant,
    polytropic_exponent,
    noise,
    threshold_rest_mass_density,
):
    l = fm_disk.angular_momentum(m, a, rmax)
    Win = fm_disk.potential(l, rin**2, 1.0, m, a)
    h = np.exp(Win - fm_disk.potential(l, r**2, sin_theta_sqrd, m, a))
    return (
        pow(
            (h - 1.0)
            * (polytropic_exponent - 1.0)
            / (polytropic_constant * polytropic_exponent),
            1.0 / (polytropic_exponent - 1.0),
        )
        - threshold_rest_mass_density
    )


def magnetic_field(
    x,
    bh_mass,
    bh_dimless_spin,
    dimless_r_in,
    dimless_r_max,
    polytropic_constant,
    polytropic_exponent,
    noise,
    threshold_density,
    plasma_beta,
):
    dummy_time = 0.0
    result = np.zeros(3)
    spin_a = bh_mass * bh_dimless_spin
    r_in = bh_mass * dimless_r_in
    r_max = bh_mass * dimless_r_max
    rho = fm_disk.rest_mass_density(
        x,
        dummy_time,
        bh_mass,
        bh_dimless_spin,
        dimless_r_in,
        dimless_r_max,
        polytropic_constant,
        polytropic_exponent,
        noise,
    )
    x_max = np.array([r_max, 0.0, 0.0])
    threshold_rho = threshold_density * (
        fm_disk.rest_mass_density(
            x_max,
            dummy_time,
            bh_mass,
            bh_dimless_spin,
            dimless_r_in,
            dimless_r_max,
            polytropic_constant,
            polytropic_exponent,
            noise,
        )
    )
    x_ks = [0.0, 0.0, 0.0]

    if rho > threshold_rho:
        small = 0.0001 * bh_mass
        a_squared = spin_a**2
        sin_theta_squared = x[0] ** 2 + x[1] ** 2
        r_squared = sin_theta_squared + x[2] ** 2
        sin_theta_squared /= r_squared

        r = np.sqrt(r_squared)
        sin_theta = np.sqrt(sin_theta_squared)
        sigma = r_squared + a_squared * x[2] ** 2 / r_squared
        prefactor = np.sqrt(sigma * (sigma + 2.0 * bh_mass * r)) * sin_theta
        prefactor = 1.0 / (2.0 * small * prefactor)
        result[0] = (
            (
                magnetic_potential(
                    r,
                    sin_theta_squared + small,
                    bh_mass,
                    spin_a,
                    r_in,
                    r_max,
                    polytropic_constant,
                    polytropic_exponent,
                    noise,
                    threshold_rho,
                )
                - magnetic_potential(
                    r,
                    sin_theta_squared - small,
                    bh_mass,
                    spin_a,
                    r_in,
                    r_max,
                    polytropic_constant,
                    polytropic_exponent,
                    noise,
                    threshold_rho,
                )
            )
            * 2.0
            * prefactor
            * sin_theta
            * x[2]
        ) / r
        result[1] = (
            magnetic_potential(
                r - small,
                sin_theta_squared,
                bh_mass,
                spin_a,
                r_in,
                r_max,
                polytropic_constant,
                polytropic_exponent,
                noise,
                threshold_rho,
            )
            - magnetic_potential(
                r + small,
                sin_theta_squared,
                bh_mass,
                spin_a,
                r_in,
                r_max,
                polytropic_constant,
                polytropic_exponent,
                noise,
                threshold_rho,
            )
        ) * prefactor
        sks_to_ks_factor = np.sqrt(r_squared + a_squared) / np.sqrt(r_squared)
        x_ks[0] = x[0] * sks_to_ks_factor
        x_ks[1] = x[1] * sks_to_ks_factor
        x_ks[2] = x[2]
        # This normalization is specific for disk parameters used
        # in test_variables function of Test_MagnetizedFmDisk.cpp.
        # Note: we are using lower (than default)
        # b_field_normalization for a faster testing time.
        normalization = 8.49943510498341
        result = normalization * ks_coords.cartesian_from_spherical_ks(
            result, x_ks, bh_mass, bh_dimless_spin
        )
        inv_jac = fm_disk.inverse_jacobian_matrix(x, spin_a)
        result = inv_jac @ result
    return result


def divergence_cleaning_field(
    x,
    bh_mass,
    bh_dimless_spin,
    dimless_r_in,
    dimless_r_max,
    polytropic_constant,
    polytropic_exponent,
    noise,
    threshold_density,
    plasma_beta,
):
    return 0.0


def lorentz_factor(
    x,
    bh_mass,
    bh_dimless_spin,
    dimless_r_in,
    dimless_r_max,
    polytropic_constant,
    polytropic_exponent,
    noise,
    threshold_density,
    plasma_beta,
):
    dummy_time = 0.0
    return fm_disk.lorentz_factor(
        x,
        dummy_time,
        bh_mass,
        bh_dimless_spin,
        dimless_r_in,
        dimless_r_max,
        polytropic_constant,
        polytropic_exponent,
        noise,
    )


def specific_enthalpy(
    x,
    bh_mass,
    bh_dimless_spin,
    dimless_r_in,
    dimless_r_max,
    polytropic_constant,
    polytropic_exponent,
    noise,
    threshold_density,
    plasma_beta,
):
    dummy_time = 0.0
    return fm_disk.specific_enthalpy(
        x,
        dummy_time,
        bh_mass,
        bh_dimless_spin,
        dimless_r_in,
        dimless_r_max,
        polytropic_constant,
        polytropic_exponent,
        noise,
    )
