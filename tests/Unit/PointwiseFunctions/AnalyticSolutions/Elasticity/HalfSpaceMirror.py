# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import scipy.integrate as integrate
from scipy.special import jv
from Elasticity.ConstitutiveRelations.IsotropicHomogeneous import (
    lame_parameter)


def alpha(k, beam_width, applied_force):
    return 1 / (2. * np.pi) * np.exp(-(k * beam_width / 2.)**2) * applied_force


def integrand_xi_w(k, w, z, lame_parameter, shear_modulus, beam_width,
                   applied_force):
    return 1 / (2. * shear_modulus) * jv(1, k * w) * np.exp(-k * z) * \
        (1 - (lame_parameter + 2. * shear_modulus) / \
        (lame_parameter + shear_modulus) + k * z) * \
        alpha(k, beam_width, applied_force)


def integrand_xi_z(k, w, z, lame_parameter, shear_modulus, beam_width,
                   applied_force):
    return 1 / (2. * shear_modulus) * jv(0, k * w) * np.exp(-k * z) * \
        (1 + shear_modulus / (lame_parameter + shear_modulus) + k * z) * \
        alpha(k, beam_width, applied_force)


def integrand_theta(k, w, z, lame_parameter, shear_modulus, beam_width,
                    applied_force):
    return 1 / (2. * shear_modulus) * k * jv(0, k * w) * np.exp(-k * z) * \
        (-2. * shear_modulus /(lame_parameter + shear_modulus)) * \
        alpha(k, beam_width, applied_force)


def integrand_strain_rz(k, w, z, lame_parameter, shear_modulus, beam_width,
                        applied_force):
    return -1 / (2. * shear_modulus) * k * jv(1, k * w) * (k * z) * \
        np.exp(-k * z) * alpha(k, beam_width, applied_force)


def integrand_strain_zz(k, w, z, lame_parameter, shear_modulus, beam_width,
                        applied_force):
    return 1 / (2. * shear_modulus) * k * jv(0, k * w) * np.exp(-k * z) * \
        (-shear_modulus /(lame_parameter + shear_modulus) - k * z) * \
        alpha(k, beam_width, applied_force)


def displacement(r, beam_width, applied_force, bulk_modulus, shear_modulus):
    local_lame_parameter = lame_parameter(bulk_modulus, shear_modulus)
    x = r[0]
    y = r[1]
    z = r[2]
    w = np.sqrt(x**2 + y**2)
    xi_w = integrate.quad(
        lambda k: integrand_xi_w(k, w, z, local_lame_parameter, shear_modulus,
                                 beam_width, applied_force),
        0,
        np.inf,
        limit=100,
        epsabs=1e-13)[0]

    xi_z = integrate.quad(
        lambda k: integrand_xi_z(k, w, z, local_lame_parameter, shear_modulus,
                                 beam_width, applied_force),
        0,
        np.inf,
        limit=100,
        epsabs=1e-13)[0]

    if w == 0.:
        assert xi_w <= 1e-13, 'xi_w is not zero at the origin'
        cos_phi = 0.
        sin_phi = 0.
    else:
        cos_phi = x / float(w)
        sin_phi = y / float(w)
    return np.asarray([xi_w * cos_phi, xi_w * sin_phi, xi_z])


def strain(r, beam_width, applied_force, bulk_modulus, shear_modulus):
    local_lame_parameter = lame_parameter(bulk_modulus, shear_modulus)

    x = r[0]
    y = r[1]
    z = r[2]
    w = np.sqrt(x**2 + y**2)

    theta = integrate.quad(
        lambda k: integrand_theta(k, w, z, local_lame_parameter, shear_modulus,
                                  beam_width, applied_force),
        0,
        np.inf,
        limit=100,
        epsabs=1e-13)[0]

    strain_rz = integrate.quad(lambda k: integrand_strain_rz(
        k, w, z, local_lame_parameter, shear_modulus, beam_width, applied_force
    ),
                               0,
                               np.inf,
                               limit=100,
                               epsabs=1e-13)[0]

    strain_zz = integrate.quad(lambda k: integrand_strain_zz(
        k, w, z, local_lame_parameter, shear_modulus, beam_width, applied_force
    ),
                               0,
                               np.inf,
                               limit=100,
                               epsabs=1e-13)[0]

    if w == 0:
        w = 1.
        strain_rr = 0.5 * (theta - strain_zz)
        strain_pp = strain_rr

    else:
        xi_w = integrate.quad(
            lambda k: integrand_xi_w(k, w, z, local_lame_parameter,
                                     shear_modulus, beam_width, applied_force),
            0,
            np.inf,
            limit=100,
            epsabs=1e-13)[0]
        strain_pp = xi_w / w
        strain_rr = theta - strain_pp - strain_zz
    return np.asarray([[
        strain_pp + (x / w)**2 * (strain_rr - strain_pp),
        x * y / w**2 * (strain_rr - strain_pp), x / w * strain_rz
    ],
                       [
                           x * y / w**2 * (strain_rr - strain_pp),
                           strain_pp + (y / w)**2 * (strain_rr - strain_pp),
                           y / w * strain_rz
                       ], [x / w * strain_rz, y / w * strain_rz, strain_zz]])


def source(r):
    return np.zeros(r.shape)
