# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import Fluxes, Sources


def tilde_d_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
                 sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                 d_lapse, d_shift, d_spatial_metric, pressure,
                 spatial_velocity, lorentz_factor, magnetic_field,
                 rest_mass_density, specific_enthalpy, extrinsic_curvature,
                 constraint_damping_parameter):
    return Fluxes.tilde_d_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
                               lapse, shift, sqrt_det_spatial_metric,
                               spatial_metric, inv_spatial_metric, pressure,
                               spatial_velocity, lorentz_factor,
                               magnetic_field)


def tilde_tau_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse,
                   shift, sqrt_det_spatial_metric, spatial_metric,
                   inv_spatial_metric, d_lapse, d_shift, d_spatial_metric,
                   pressure, spatial_velocity, lorentz_factor, magnetic_field,
                   rest_mass_density, specific_enthalpy, extrinsic_curvature,
                   constraint_damping_parameter):
    return Fluxes.tilde_tau_flux(tilde_d, tilde_tau, tilde_s, tilde_b,
                                 tilde_phi, lapse, shift,
                                 sqrt_det_spatial_metric, spatial_metric,
                                 inv_spatial_metric, pressure,
                                 spatial_velocity, lorentz_factor,
                                 magnetic_field)


def tilde_s_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
                 sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                 d_lapse, d_shift, d_spatial_metric, pressure,
                 spatial_velocity, lorentz_factor, magnetic_field,
                 rest_mass_density, specific_enthalpy, extrinsic_curvature,
                 constraint_damping_parameter):
    return Fluxes.tilde_s_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
                               lapse, shift, sqrt_det_spatial_metric,
                               spatial_metric, inv_spatial_metric, pressure,
                               spatial_velocity, lorentz_factor,
                               magnetic_field)


def tilde_b_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
                 sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                 d_lapse, d_shift, d_spatial_metric, pressure,
                 spatial_velocity, lorentz_factor, magnetic_field,
                 rest_mass_density, specific_enthalpy, extrinsic_curvature,
                 constraint_damping_parameter):
    return Fluxes.tilde_b_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi,
                               lapse, shift, sqrt_det_spatial_metric,
                               spatial_metric, inv_spatial_metric, pressure,
                               spatial_velocity, lorentz_factor,
                               magnetic_field)


def tilde_phi_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse,
                   shift, sqrt_det_spatial_metric, spatial_metric,
                   inv_spatial_metric, d_lapse, d_shift, d_spatial_metric,
                   pressure, spatial_velocity, lorentz_factor, magnetic_field,
                   rest_mass_density, specific_enthalpy, extrinsic_curvature,
                   constraint_damping_parameter):
    return Fluxes.tilde_phi_flux(tilde_d, tilde_tau, tilde_s, tilde_b,
                                 tilde_phi, lapse, shift,
                                 sqrt_det_spatial_metric, spatial_metric,
                                 inv_spatial_metric, pressure,
                                 spatial_velocity, lorentz_factor,
                                 magnetic_field)


def source_tilde_d(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse,
                   shift, sqrt_det_spatial_metric, spatial_metric,
                   inv_spatial_metric, d_lapse, d_shift, d_spatial_metric,
                   pressure, spatial_velocity, lorentz_factor, magnetic_field,
                   rest_mass_density, specific_enthalpy, extrinsic_curvature,
                   constraint_damping_parameter):
    return 0.0 * lapse


def source_tilde_tau(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse,
                     shift, sqrt_det_spatial_metric, spatial_metric,
                     inv_spatial_metric, d_lapse, d_shift, d_spatial_metric,
                     pressure, spatial_velocity, lorentz_factor,
                     magnetic_field, rest_mass_density, specific_enthalpy,
                     extrinsic_curvature, constraint_damping_parameter):
    return Sources.source_tilde_tau(
        tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, spatial_velocity,
        magnetic_field, rest_mass_density, specific_enthalpy, lorentz_factor,
        pressure, lapse, d_lapse, d_shift, spatial_metric, d_spatial_metric,
        inv_spatial_metric, sqrt_det_spatial_metric, extrinsic_curvature,
        constraint_damping_parameter)


def source_tilde_s(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse,
                   shift, sqrt_det_spatial_metric, spatial_metric,
                   inv_spatial_metric, d_lapse, d_shift, d_spatial_metric,
                   pressure, spatial_velocity, lorentz_factor, magnetic_field,
                   rest_mass_density, specific_enthalpy, extrinsic_curvature,
                   constraint_damping_parameter):
    return Sources.source_tilde_s(
        tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, spatial_velocity,
        magnetic_field, rest_mass_density, specific_enthalpy, lorentz_factor,
        pressure, lapse, d_lapse, d_shift, spatial_metric, d_spatial_metric,
        inv_spatial_metric, sqrt_det_spatial_metric, extrinsic_curvature,
        constraint_damping_parameter)


def source_tilde_b(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse,
                   shift, sqrt_det_spatial_metric, spatial_metric,
                   inv_spatial_metric, d_lapse, d_shift, d_spatial_metric,
                   pressure, spatial_velocity, lorentz_factor, magnetic_field,
                   rest_mass_density, specific_enthalpy, extrinsic_curvature,
                   constraint_damping_parameter):
    return Sources.source_tilde_b(
        tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, spatial_velocity,
        magnetic_field, rest_mass_density, specific_enthalpy, lorentz_factor,
        pressure, lapse, d_lapse, d_shift, spatial_metric, d_spatial_metric,
        inv_spatial_metric, sqrt_det_spatial_metric, extrinsic_curvature,
        constraint_damping_parameter)


def source_tilde_phi(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse,
                     shift, sqrt_det_spatial_metric, spatial_metric,
                     inv_spatial_metric, d_lapse, d_shift, d_spatial_metric,
                     pressure, spatial_velocity, lorentz_factor,
                     magnetic_field, rest_mass_density, specific_enthalpy,
                     extrinsic_curvature, constraint_damping_parameter):
    return Sources.source_tilde_phi(
        tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, spatial_velocity,
        magnetic_field, rest_mass_density, specific_enthalpy, lorentz_factor,
        pressure, lapse, d_lapse, d_shift, spatial_metric, d_spatial_metric,
        inv_spatial_metric, sqrt_det_spatial_metric, extrinsic_curvature,
        constraint_damping_parameter)
