# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

import PointwiseFunctions.GeneralRelativity.KerrSchildCoords as ks_coords


def rest_mass_density(x, bh_mass, bh_dimless_spin, rest_mass_density,
                      flow_speed, mag_field_strength, polytropic_constant,
                      polytropic_exponent):
    return rest_mass_density


def spatial_velocity(x, bh_mass, bh_dimless_spin, rest_mass_density,
                     flow_speed, mag_field_strength, polytropic_constant,
                     polytropic_exponent):
    result = np.zeros(3)
    spin_a = bh_mass * bh_dimless_spin
    a_squared = spin_a**2
    r_squared = ks_coords.r_coord_squared(x, bh_mass, bh_dimless_spin)
    cos_theta = x[2] / np.sqrt(r_squared)
    cos_theta_squared = cos_theta**2
    sigma = r_squared + a_squared * cos_theta_squared

    result[0] = (flow_speed * cos_theta /
                 np.sqrt(1.0 + 2.0 * bh_mass * np.sqrt(r_squared) /
                         sigma))
    result[1] = - flow_speed * \
        np.sqrt(1.0 - cos_theta_squared) / np.sqrt(sigma)

    return ks_coords.cartesian_from_spherical_ks(result, x, bh_mass,
                                                 bh_dimless_spin)


def specific_internal_energy(x, bh_mass, bh_dimless_spin, rest_mass_density,
                             flow_speed, mag_field_strength,
                             polytropic_constant, polytropic_exponent):
    return (polytropic_constant *
            pow(rest_mass_density, polytropic_exponent - 1.0) /
            (polytropic_exponent - 1.0))


def pressure(x, bh_mass, bh_dimless_spin, rest_mass_density, flow_speed,
             mag_field_strength, polytropic_constant, polytropic_exponent):
    return polytropic_constant * pow(rest_mass_density, polytropic_exponent)


def magnetic_field(x, bh_mass, bh_dimless_spin, rest_mass_density, flow_speed,
                   mag_field_strength, polytropic_constant,
                   polytropic_exponent):
    result = np.zeros(3)
    spin_a = bh_mass * bh_dimless_spin
    a_squared = spin_a**2
    r_squared = ks_coords.r_coord_squared(x, bh_mass, bh_dimless_spin)
    cos_theta = x[2] / np.sqrt(r_squared)
    cos_theta_squared = cos_theta**2
    two_m_r = 2.0 * bh_mass * np.sqrt(r_squared)
    sigma = r_squared + a_squared * cos_theta_squared

    result[0:] = mag_field_strength / np.sqrt(sigma * (sigma + two_m_r))
    result[0] *= (r_squared - two_m_r + a_squared + two_m_r *
                  (r_squared**2 - a_squared**2) / sigma**2) * cos_theta
    result[1] *= -((np.sqrt(r_squared) +
                    bh_mass * a_squared *
                    (r_squared - a_squared * cos_theta_squared) *
                    (1.0 + cos_theta_squared) / sigma**2) *
                   np.sqrt(1.0 - cos_theta_squared))
    result[2] *= spin_a * (1.0 + two_m_r * (r_squared - a_squared) /
                           sigma**2) * cos_theta
    return ks_coords.cartesian_from_spherical_ks(result, x, bh_mass,
                                                 bh_dimless_spin)


def divergence_cleaning_field(x, bh_mass, bh_dimless_spin, rest_mass_density,
                              flow_speed, mag_field_strength,
                              polytropic_constant, polytropic_exponent):
    return 0.0


def lorentz_factor(x, bh_mass, bh_dimless_spin, rest_mass_density, flow_speed,
                   mag_field_strength, polytropic_constant,
                   polytropic_exponent):
    return 1.0 / np.sqrt(1.0 - flow_speed**2)


def specific_enthalpy(x, bh_mass, bh_dimless_spin, rest_mass_density,
                      flow_speed, mag_field_strength, polytropic_constant,
                      polytropic_exponent):
    return (1.0 + (polytropic_constant * polytropic_exponent *
                   pow(rest_mass_density, polytropic_exponent - 1.0) /
                   (polytropic_exponent - 1.0)))
