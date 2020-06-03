# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import scipy.integrate as integrate
from scipy.special import jv
from Elasticity.ConstitutiveRelations.IsotropicHomogeneous import (
    lame_parameter)


def beam_profile(k, beam_width):
    return (1. / (2. * np.pi) * np.exp(-(k * beam_width / 2.)**2))


def integrand_xi_w(k, w, z, lame_parameter, shear_modulus, beam_width):
    return (1. / (2. * shear_modulus) * jv(1, k * w) * np.exp(-k * z) *
            (1. - (lame_parameter + 2. * shear_modulus) /
             (lame_parameter + shear_modulus) + k * z) *
            beam_profile(k, beam_width))


def integrand_xi_z(k, w, z, lame_parameter, shear_modulus, beam_width):
    return (1. / (2. * shear_modulus) * jv(0, k * w) * np.exp(-k * z) *
            (1. + shear_modulus / (lame_parameter + shear_modulus) + k * z) *
            beam_profile(k, beam_width))


def integrand_theta(k, w, z, lame_parameter, shear_modulus, beam_width):
    return (1. / (2. * shear_modulus) * k * jv(0, k * w) * np.exp(-k * z) *
            (-2. * shear_modulus /
             (lame_parameter + shear_modulus)) * beam_profile(k, beam_width))


def integrand_strain_rz(k, w, z, lame_parameter, shear_modulus, beam_width):
    return (-1. / (2. * shear_modulus) * k * jv(1, k * w) * (k * z) *
            np.exp(-k * z) * beam_profile(k, beam_width))


def integrand_strain_zz(k, w, z, lame_parameter, shear_modulus, beam_width):
    return (1. / (2. * shear_modulus) * k * jv(0, k * w) * np.exp(-k * z) *
            (-shear_modulus / (lame_parameter + shear_modulus) - k * z) *
            beam_profile(k, beam_width))


def displacement(r, beam_width, bulk_modulus, shear_modulus):
    local_lame_parameter = lame_parameter(bulk_modulus, shear_modulus)
    x = r[0]
    y = r[1]
    z = r[2]
    radius = np.sqrt(x**2 + y**2)
    xi_z = integrate.quad(lambda k: integrand_xi_z(
        k, radius, z, local_lame_parameter, shear_modulus, beam_width),
                          0,
                          np.inf,
                          limit=100,
                          epsabs=1e-13)[0]
    if radius <= 1e-13:
        return np.asarray(0., 0., xi_z)
    xi_w = integrate.quad(lambda k: integrand_xi_w(
        k, radius, z, local_lame_parameter, shear_modulus, beam_width),
                          0,
                          np.inf,
                          limit=100,
                          epsabs=1e-13)[0]

    cos_phi = x / radius
    sin_phi = y / radius
    return np.asarray([xi_w * cos_phi, xi_w * sin_phi, xi_z])


def strain(r, beam_width, bulk_modulus, shear_modulus):
    local_lame_parameter = lame_parameter(bulk_modulus, shear_modulus)

    x = r[0]
    y = r[1]
    z = r[2]
    radius = np.sqrt(x**2 + y**2)

    trace_term = integrate.quad(lambda k: integrand_theta(
        k, radius, z, local_lame_parameter, shear_modulus, beam_width),
                                0,
                                np.inf,
                                limit=100,
                                epsabs=1e-13)[0]

    strain_zz = integrate.quad(lambda k: integrand_strain_zz(
        k, radius, z, local_lame_parameter, shear_modulus, beam_width),
                               0,
                               np.inf,
                               limit=100,
                               epsabs=1e-13)[0]

    if radius <= 1e-13:
        radius = 1.
        strain_rr = 0.5 * (trace_term - strain_zz)
        return np.asarray([[strain_rr, 0, 0], [0, strain_rr, 0],
                           [0, 0, strain_zz]])
    else:
        strain_rz = integrate.quad(lambda k: integrand_strain_rz(
            k, radius, z, local_lame_parameter, shear_modulus, beam_width),
                                   0,
                                   np.inf,
                                   limit=100,
                                   epsabs=1e-13)[0]
        xi_w = integrate.quad(lambda k: integrand_xi_w(
            k, radius, z, local_lame_parameter, shear_modulus, beam_width),
                              0,
                              np.inf,
                              limit=100,
                              epsabs=1e-13)[0]
        strain_pp = xi_w / radius
        strain_rr = trace_term - strain_pp - strain_zz
        return np.asarray(
            [[
                strain_pp + (x / radius)**2 * (strain_rr - strain_pp),
                x * y / radius**2 * (strain_rr - strain_pp),
                x / radius * strain_rz
            ],
             [
                 x * y / radius**2 * (strain_rr - strain_pp),
                 strain_pp + (y / radius)**2 * (strain_rr - strain_pp),
                 y / radius * strain_rz
             ], [x / radius * strain_rz, y / radius * strain_rz, strain_zz]])


def source(r):
    return np.zeros(r.shape)
