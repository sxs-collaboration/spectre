# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def gauge_wave_h(x, t, amplitude, wavelength):
    omega = 2.0 * np.pi / wavelength
    return 1.0 - amplitude * np.sin(omega * (x[0] - t))


def gauge_wave_dh(x, t, amplitude, wavelength):
    omega = 2.0 * np.pi / wavelength
    return -1.0 * omega * amplitude * np.cos(omega * (x[0] - t))


def gauge_wave_sqrt_h(x, t, amplitude, wavelength):
    return np.sqrt(gauge_wave_h(x, t, amplitude, wavelength))


def gauge_wave_dh_over_2_sqrt_h(x, t, amplitude, wavelength):
    dh = gauge_wave_dh(x, t, amplitude, wavelength)
    sqrt_h = gauge_wave_sqrt_h(x, t, amplitude, wavelength)
    return 0.5 * dh / sqrt_h


def gauge_wave_lapse(x, t, amplitude, wavelength):
    return gauge_wave_sqrt_h(x, t, amplitude, wavelength)


def gauge_wave_dt_lapse(x, t, amplitude, wavelength):
    return -1.0 * gauge_wave_dh_over_2_sqrt_h(x, t, amplitude, wavelength)


def gauge_wave_d_lapse(x, t, amplitude, wavelength):
    d_lapse = np.zeros_like(x)
    d_lapse[0] = gauge_wave_dh_over_2_sqrt_h(x, t, amplitude, wavelength)
    return d_lapse


def gauge_wave_shift(x, t, amplitude, wavelength):
    return np.zeros_like(x)


def gauge_wave_dt_shift(x, t, amplitude, wavelength):
    return np.zeros_like(x)


def gauge_wave_d_shift(x, t, amplitude, wavelength):
    return np.zeros((len(x), len(x)))


def gauge_wave_spatial_metric(x, t, amplitude, wavelength):
    spatial_metric = np.zeros((len(x), len(x)))
    spatial_metric[0, 0] = gauge_wave_h(x, t, amplitude, wavelength)
    spatial_metric[1, 1] = 1.0
    spatial_metric[2, 2] = 1.0
    return spatial_metric


def gauge_wave_dt_spatial_metric(x, t, amplitude, wavelength):
    dt_spatial_metric = np.zeros((len(x), len(x)))
    dt_spatial_metric[0, 0] = -1.0 * gauge_wave_dh(x, t, amplitude, wavelength)
    return dt_spatial_metric


def gauge_wave_d_spatial_metric(x, t, amplitude, wavelength):
    d_spatial_metric = np.zeros((len(x), len(x), len(x)))
    d_spatial_metric[0, 0, 0] = gauge_wave_dh(x, t, amplitude, wavelength)
    return d_spatial_metric


def gauge_wave_sqrt_det_spatial_metric(x, t, amplitude, wavelength):
    return gauge_wave_sqrt_h(x, t, amplitude, wavelength)


def gauge_wave_extrinsic_curvature(x, t, amplitude, wavelength):
    extrinsic_curvature = np.zeros((len(x), len(x)))
    extrinsic_curvature[0, 0] = gauge_wave_dh_over_2_sqrt_h(
        x, t, amplitude, wavelength)
    return extrinsic_curvature


def gauge_wave_inverse_spatial_metric(x, t, amplitude, wavelength):
    inverse_spatial_metric = np.zeros((len(x), len(x)))
    inverse_spatial_metric[0,
                           0] = 1.0 / gauge_wave_h(x, t, amplitude, wavelength)
    inverse_spatial_metric[1, 1] = 1.0
    inverse_spatial_metric[2, 2] = 1.0
    return inverse_spatial_metric
