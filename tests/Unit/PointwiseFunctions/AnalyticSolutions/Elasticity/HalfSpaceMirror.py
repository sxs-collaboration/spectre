# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import scipy.integrate as integrate
from scipy.special import jv


def alpha(k, shear_modulus, beam_width, applied_force):
    return 1 / (4. * np.pi * shear_modulus
                ) * np.exp(-(k * beam_width / 2.)**2) * applied_force


def integrand_xi_w(k, w, z, poisson_ratio, shear_modulus, beam_width,
                   applied_force):
    return (-1 + 2.*poisson_ratio + k*z)*np.exp(-k*z)*jv(1, k*w) * \
        alpha(k, shear_modulus, beam_width, applied_force)


def integrand_xi_z(k, w, z, poisson_ratio, shear_modulus, beam_width,
                   applied_force):
    return (2 - 2.*poisson_ratio + k*z)*np.exp(-k*z)*jv(0, k*w) * \
        alpha(k, shear_modulus, beam_width, applied_force)


def integrand_xi_w_dw(k, w, z, poisson_ratio, shear_modulus, beam_width,
                      applied_force):
    return (-1 + 2. * poisson_ratio + k * z) * np.exp(
        -k * z) * k * (jv(0, k * w) - jv(2, k * w)) / 2. * alpha(
            k, shear_modulus, beam_width, applied_force)


def integrand_xi_w_dz(k, w, z, poisson_ratio, shear_modulus, beam_width,
                      applied_force):
    return (2 - 2.*poisson_ratio - k*z)*k*np.exp(-k*z)*jv(1, k*w) * \
        alpha(k, shear_modulus, beam_width, applied_force)


def integrand_xi_z_dw(k, w, z, poisson_ratio, shear_modulus, beam_width,
                      applied_force):
    return (2 - 2.*poisson_ratio + k*z)*np.exp(-k*z)*k*(-jv(1, k*w)) * \
        alpha(k, shear_modulus, beam_width, applied_force)


def integrand_xi_z_dz(k, w, z, poisson_ratio, shear_modulus, beam_width,
                      applied_force):
    return (-1 + 2.*poisson_ratio - k*z)*k*np.exp(-k*z)*jv(0, k*w) * \
        alpha(k, shear_modulus, beam_width, applied_force)


def displacement(r, beam_width, applied_force, bulk_modulus, shear_modulus):
    poisson_ratio = (3. * bulk_modulus - 2. * shear_modulus) / \
        (6. * bulk_modulus + 2. * shear_modulus)

    x = r[0]
    y = r[1]
    z = r[2]
    w = np.sqrt(x**2 + y**2)
    xi_w = integrate.quad(lambda k: integrand_xi_w(
        k, w, z, poisson_ratio, shear_modulus, beam_width, applied_force),
                          0,
                          np.inf,
                          limit=100,
                          epsabs=1e-15)[0]
    xi_z = integrate.quad(lambda k: integrand_xi_z(
        k, w, z, poisson_ratio, shear_modulus, beam_width, applied_force),
                          0,
                          np.inf,
                          limit=100,
                          epsabs=1e-15)[0]
    if w == 0.:
        assert xi_w <= 1e-13, 'xi_w is not zero at the origin'
        cos_phi = 0.
        sin_phi = 0.
    else:
        cos_phi = x / float(w)
        sin_phi = y / float(w)
    return np.asarray([xi_w * cos_phi, xi_w * sin_phi, xi_z])


def strain(r, beam_width, applied_force, bulk_modulus, shear_modulus):
    poisson_ratio = (3. * bulk_modulus - 2. * shear_modulus) / \
        (6. * bulk_modulus + 2. * shear_modulus)

    x = r[0]
    y = r[1]
    z = r[2]
    w = np.sqrt(x**2 + y**2)

    xi_z_dw = integrate.quad(lambda k: integrand_xi_z_dw(
        k, w, z, poisson_ratio, shear_modulus, beam_width, applied_force),
                             0,
                             np.inf,
                             limit=100,
                             epsabs=1e-15)[0]

    xi_w_dz = integrate.quad(lambda k: integrand_xi_w_dz(
        k, w, z, poisson_ratio, shear_modulus, beam_width, applied_force),
                             0,
                             np.inf,
                             limit=100,
                             epsabs=1e-15)[0]

    xi_z_dz = integrate.quad(lambda k: integrand_xi_z_dz(
        k, w, z, poisson_ratio, shear_modulus, beam_width, applied_force),
                             0,
                             np.inf,
                             limit=100,
                             epsabs=1e-15)[0]

    xi_w_dw = integrate.quad(lambda k: integrand_xi_w_dw(
        k, w, z, poisson_ratio, shear_modulus, beam_width, applied_force),
                             0,
                             np.inf,
                             limit=100,
                             epsabs=1e-15)[0]
    if w == 0:
        w = 1.
        xi_w_over_w = xi_w_dw
    else:
        xi_w_over_w = integrate.quad(lambda k: integrand_xi_w(
            k, w, z, poisson_ratio, shear_modulus, beam_width, applied_force),
                                     0,
                                     np.inf,
                                     limit=100,
                                     epsabs=1e-15)[0] / w
    return np.asarray([[
        (y / w)**2 * xi_w_over_w + (x / w)**2 * xi_w_dw,
        x * y / w**2 * (xi_w_dw - xi_w_over_w),
        x / w * (xi_w_dz + xi_z_dw) / 2.
    ],
    [
        x * y / w**2 * (xi_w_dw - xi_w_over_w),
        (x / w)**2 * xi_w_over_w + (y / w)**2 * xi_w_dw,
        y / w * (xi_w_dz + xi_z_dw) / 2.
    ],
    [
        x / w * (xi_w_dz + xi_z_dw) / 2.,
        y / w * (xi_w_dz + xi_z_dw) / 2., xi_z_dz
    ]])


def source(r):
    return np.zeros(r.shape)
