# Distributed under the MIT License.
# See LICENSE.txt for details.

import Evolution.Systems.RadiationTransport.M1Grey.Fluxes as fluxes
import numpy as np
import PointwiseFunctions.AnalyticData.RadiationTransport.M1Grey.HomogeneousSphere as data
import PointwiseFunctions.AnalyticSolutions.RadiationTransport.M1Grey.ConstantM1 as soln


def soln_error(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
):
    return None


# constantM1 data
_soln_mean_velocity = np.array([0.1, 0.2, 0.3])
_soln_comoving_energy_density = 0.4

# There is no Python implementation of the M1Closure, so for now we hardcode
# the pressure tensor with values obtained by running the C++ code run with
# the same input parameters. This is hacky, because it couples the C++ and
# Python implementations and makes the test less robust.
_tilde_p_values_from_cxx = np.array(
    [
        [0.13953488372093026, 0.012403100775193798, 0.018604651162790694],
        [0.012403100775193798, 0.15813953488372096, 0.037209302325581388],
        [0.018604651162790694, 0.037209302325581388, 0.18914728682170545],
    ]
)

_radius = 1.0
_emissivity_and_opacity = 1.0
_outer_radius = 1.5
_outer_opacity = 0.5
_boundary_roundness = 0.03


def minerbo_closure(zeta):
    return 1.0 / 3.0 + zeta * zeta * (
        0.4 - 2.0 / 15.0 * zeta + 0.4 * zeta * zeta
    )


# compute pressure tensor for zero/low velocity case
# non-zero velocities still have to be implemented
def calc_tilde_p(energy_density, momentum_density, inverse_spatial_metric):
    contravariant_momentum_density = np.einsum(
        "i, ij", momentum_density, inverse_spatial_metric
    )

    small_number = 1.0e-150

    s2 = max(
        (np.dot(momentum_density, contravariant_momentum_density), small_number)
    )

    zeta = np.sqrt(s2) / energy_density

    chi = minerbo_closure(zeta)

    d_thin_energy_over_momentum_squared = (
        (1.5 * chi - 0.5) * energy_density / s2
    )

    d_thick_energy_over_3 = (0.5 - 0.5 * chi) * energy_density

    dimensions = [0, 1, 2]

    pressure_tensor = np.zeros((3, 3))

    for i in dimensions:
        for j in dimensions:
            pressure_tensor[i, j] = (
                d_thick_energy_over_3 * inverse_spatial_metric[i, j]
                + d_thin_energy_over_momentum_squared
                * contravariant_momentum_density[i]
                * contravariant_momentum_density[j]
            )

    return pressure_tensor


def soln_tilde_e_nue_const(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
):
    return soln.tildeE(
        coords, time, _soln_mean_velocity, _soln_comoving_energy_density
    )


def soln_tilde_e_nue(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
):
    return data.m1_tildeE(
        coords,
        _radius,
        _emissivity_and_opacity,
        _outer_opacity,
        _boundary_roundness,
    )


def soln_tilde_e_bar_nue_const(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
):
    # same as nue
    return soln_tilde_e_nue_const(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
    )


def soln_tilde_e_bar_nue(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
):
    # same as nue
    return soln_tilde_e_nue(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
    )


def soln_tilde_s_nue_const(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
):
    return soln.tildeS(
        coords, time, _soln_mean_velocity, _soln_comoving_energy_density
    )


def soln_tilde_s_nue(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
):
    return data.m1_tildeS(
        coords,
        _radius,
        _emissivity_and_opacity,
        _outer_opacity,
        _boundary_roundness,
    )


# same as nue
def soln_tilde_s_bar_nue_const(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
):
    return soln_tilde_s_nue_const(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
    )


def soln_tilde_s_bar_nue(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
):
    return soln_tilde_s_nue(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
    )


def soln_flux_tilde_e_nue_const(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
):
    tilde_e = soln_tilde_e_nue_const(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
    )
    tilde_s = soln_tilde_s_nue_const(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
    )
    lapse = 1.0
    shift = np.array([0.0, 0.0, 0.0])
    spatial_metric = np.identity(3)
    inv_spatial_metric = np.identity(3)
    tilde_p = _tilde_p_values_from_cxx

    return fluxes.tilde_e_flux(
        tilde_e,
        tilde_s,
        tilde_p,
        lapse,
        shift,
        spatial_metric,
        inv_spatial_metric,
    )


def soln_flux_tilde_e_nue(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
):
    tilde_e = soln_tilde_e_nue(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
    )
    tilde_s = soln_tilde_s_nue(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
    )
    lapse = 1.0
    shift = np.array([0.0, 0.0, 0.0])
    spatial_metric = np.identity(3)
    inv_spatial_metric = np.identity(3)
    tilde_p = calc_tilde_p(tilde_e, tilde_s, inv_spatial_metric)

    return fluxes.tilde_e_flux(
        tilde_e,
        tilde_s,
        tilde_p,
        lapse,
        shift,
        spatial_metric,
        inv_spatial_metric,
    )


# same as nue
def soln_flux_tilde_e_bar_nue_const(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
):
    return soln_flux_tilde_e_nue_const(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
    )


def soln_flux_tilde_e_bar_nue(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
):
    return soln_flux_tilde_e_nue(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
    )


def soln_flux_tilde_s_nue_const(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
):
    tilde_e = soln_tilde_e_nue_const(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
    )
    tilde_s = soln_tilde_s_nue_const(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
    )
    lapse = 1.0
    shift = np.array([0.0, 0.0, 0.0])
    spatial_metric = np.identity(3)
    inv_spatial_metric = np.identity(3)
    tilde_p = _tilde_p_values_from_cxx

    return fluxes.tilde_s_flux(
        tilde_e,
        tilde_s,
        tilde_p,
        lapse,
        shift,
        spatial_metric,
        inv_spatial_metric,
    )


def soln_flux_tilde_s_nue(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
):
    tilde_e = soln_tilde_e_nue(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
    )
    tilde_s = soln_tilde_s_nue(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
    )
    lapse = 1.0
    shift = np.array([0.0, 0.0, 0.0])
    spatial_metric = np.identity(3)
    inv_spatial_metric = np.identity(3)
    tilde_p = calc_tilde_p(tilde_e, tilde_s, inv_spatial_metric)

    return fluxes.tilde_s_flux(
        tilde_e,
        tilde_s,
        tilde_p,
        lapse,
        shift,
        spatial_metric,
        inv_spatial_metric,
    )


# same as nue
def soln_flux_tilde_s_bar_nue_const(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
):
    return soln_flux_tilde_s_nue_const(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
    )


def soln_flux_tilde_s_bar_nue(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
):
    return soln_flux_tilde_s_nue(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
    )
