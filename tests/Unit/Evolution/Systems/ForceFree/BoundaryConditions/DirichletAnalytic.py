# Distributed under the MIT License.
# See LICENSE.txt for details.

import Evolution.Systems.ForceFree.ElectricCurrentDensity as current
import Evolution.Systems.ForceFree.Fluxes as fluxes
import numpy as np
import PointwiseFunctions.AnalyticData.ForceFree.FfeBreakdown as breakdown
import PointwiseFunctions.AnalyticSolutions.ForceFree.FastWave as fastwave


def soln_error(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    return None


def soln_lapse(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    return 1.0


def soln_shift(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    return np.asarray([0.0, 0.0, 0.0])


def soln_sqrt_det_spatial_metric(coords, time):
    return 1.0


def soln_spatial_metric(coords, time):
    return np.identity(3)


def soln_inverse_spatial_metric(coords, time):
    return np.identity(3)


def soln_tilde_e(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    return fastwave.TildeE(coords, time)


def soln_tilde_b(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    return fastwave.TildeB(coords, time)


def soln_tilde_psi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    return fastwave.TildePsi(coords, time)


def soln_tilde_phi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    return fastwave.TildePhi(coords, time)


def soln_tilde_q(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    return fastwave.TildeQ(coords, time)


def soln_flux_tilde_e(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    tilde_e = fastwave.TildeE(coords, time)
    tilde_b = fastwave.TildeB(coords, time)
    tilde_psi = fastwave.TildePsi(coords, time)
    tilde_phi = fastwave.TildePhi(coords, time)
    tilde_q = fastwave.TildeQ(coords, time)
    lapse = soln_lapse(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    shift = soln_shift(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    sqrt_det_spatial_metric = soln_sqrt_det_spatial_metric(coords, time)
    spatial_metric = soln_spatial_metric(coords, time)
    inv_spatial_metric = soln_inverse_spatial_metric(coords, time)

    return fluxes.tilde_e_flux(
        tilde_e,
        tilde_b,
        tilde_psi,
        tilde_phi,
        tilde_q,
        current.tilde_j(
            tilde_q,
            tilde_e,
            tilde_b,
            parallel_conductivity,
            lapse,
            sqrt_det_spatial_metric,
            spatial_metric,
        ),
        lapse,
        shift,
        sqrt_det_spatial_metric,
        spatial_metric,
        inv_spatial_metric,
    )


def soln_flux_tilde_b(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    tilde_e = fastwave.TildeE(coords, time)
    tilde_b = fastwave.TildeB(coords, time)
    tilde_psi = fastwave.TildePsi(coords, time)
    tilde_phi = fastwave.TildePhi(coords, time)
    tilde_q = fastwave.TildeQ(coords, time)
    lapse = soln_lapse(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    shift = soln_shift(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    sqrt_det_spatial_metric = soln_sqrt_det_spatial_metric(coords, time)
    spatial_metric = soln_spatial_metric(coords, time)
    inv_spatial_metric = soln_inverse_spatial_metric(coords, time)

    return fluxes.tilde_b_flux(
        tilde_e,
        tilde_b,
        tilde_psi,
        tilde_phi,
        tilde_q,
        current.tilde_j(
            tilde_q,
            tilde_e,
            tilde_b,
            parallel_conductivity,
            lapse,
            sqrt_det_spatial_metric,
            spatial_metric,
        ),
        lapse,
        shift,
        sqrt_det_spatial_metric,
        spatial_metric,
        inv_spatial_metric,
    )


def soln_flux_tilde_psi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    tilde_e = fastwave.TildeE(coords, time)
    tilde_b = fastwave.TildeB(coords, time)
    tilde_psi = fastwave.TildePsi(coords, time)
    tilde_phi = fastwave.TildePhi(coords, time)
    tilde_q = fastwave.TildeQ(coords, time)
    lapse = soln_lapse(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    shift = soln_shift(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    sqrt_det_spatial_metric = soln_sqrt_det_spatial_metric(coords, time)
    spatial_metric = soln_spatial_metric(coords, time)
    inv_spatial_metric = soln_inverse_spatial_metric(coords, time)

    return fluxes.tilde_psi_flux(
        tilde_e,
        tilde_b,
        tilde_psi,
        tilde_phi,
        tilde_q,
        current.tilde_j(
            tilde_q,
            tilde_e,
            tilde_b,
            parallel_conductivity,
            lapse,
            sqrt_det_spatial_metric,
            spatial_metric,
        ),
        lapse,
        shift,
        sqrt_det_spatial_metric,
        spatial_metric,
        inv_spatial_metric,
    )


def soln_flux_tilde_phi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    tilde_e = fastwave.TildeE(coords, time)
    tilde_b = fastwave.TildeB(coords, time)
    tilde_psi = fastwave.TildePsi(coords, time)
    tilde_phi = fastwave.TildePhi(coords, time)
    tilde_q = fastwave.TildeQ(coords, time)
    lapse = soln_lapse(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    shift = soln_shift(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    sqrt_det_spatial_metric = soln_sqrt_det_spatial_metric(coords, time)
    spatial_metric = soln_spatial_metric(coords, time)
    inv_spatial_metric = soln_inverse_spatial_metric(coords, time)

    return fluxes.tilde_phi_flux(
        tilde_e,
        tilde_b,
        tilde_psi,
        tilde_phi,
        tilde_q,
        current.tilde_j(
            tilde_q,
            tilde_e,
            tilde_b,
            parallel_conductivity,
            lapse,
            sqrt_det_spatial_metric,
            spatial_metric,
        ),
        lapse,
        shift,
        sqrt_det_spatial_metric,
        spatial_metric,
        inv_spatial_metric,
    )


def soln_flux_tilde_q(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    tilde_e = fastwave.TildeE(coords, time)
    tilde_b = fastwave.TildeB(coords, time)
    tilde_psi = fastwave.TildePsi(coords, time)
    tilde_phi = fastwave.TildePhi(coords, time)
    tilde_q = fastwave.TildeQ(coords, time)
    lapse = soln_lapse(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    shift = soln_shift(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    sqrt_det_spatial_metric = soln_sqrt_det_spatial_metric(coords, time)
    spatial_metric = soln_spatial_metric(coords, time)
    inv_spatial_metric = soln_inverse_spatial_metric(coords, time)

    return fluxes.tilde_q_flux(
        tilde_e,
        tilde_b,
        tilde_psi,
        tilde_phi,
        tilde_q,
        current.tilde_j(
            tilde_q,
            tilde_e,
            tilde_b,
            parallel_conductivity,
            lapse,
            sqrt_det_spatial_metric,
            spatial_metric,
        ),
        lapse,
        shift,
        sqrt_det_spatial_metric,
        spatial_metric,
        inv_spatial_metric,
    )


def data_tilde_e(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    return breakdown.TildeE(coords)


def data_tilde_b(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    return breakdown.TildeB(coords)


def data_tilde_psi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    return breakdown.TildePsi(coords)


def data_tilde_phi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    return breakdown.TildePhi(coords)


def data_tilde_q(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    return breakdown.TildeQ(coords)


def data_flux_tilde_e(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    tilde_e = breakdown.TildeE(coords)
    tilde_b = breakdown.TildeB(coords)
    tilde_psi = breakdown.TildePsi(coords)
    tilde_phi = breakdown.TildePhi(coords)
    tilde_q = breakdown.TildeQ(coords)
    lapse = soln_lapse(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    shift = soln_shift(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    sqrt_det_spatial_metric = soln_sqrt_det_spatial_metric(coords, time)
    spatial_metric = soln_spatial_metric(coords, time)
    inv_spatial_metric = soln_inverse_spatial_metric(coords, time)

    return fluxes.tilde_e_flux(
        tilde_e,
        tilde_b,
        tilde_psi,
        tilde_phi,
        tilde_q,
        current.tilde_j(
            tilde_q,
            tilde_e,
            tilde_b,
            parallel_conductivity,
            lapse,
            sqrt_det_spatial_metric,
            spatial_metric,
        ),
        lapse,
        shift,
        sqrt_det_spatial_metric,
        spatial_metric,
        inv_spatial_metric,
    )


def data_flux_tilde_b(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    tilde_e = breakdown.TildeE(coords)
    tilde_b = breakdown.TildeB(coords)
    tilde_psi = breakdown.TildePsi(coords)
    tilde_phi = breakdown.TildePhi(coords)
    tilde_q = breakdown.TildeQ(coords)
    lapse = soln_lapse(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    shift = soln_shift(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    sqrt_det_spatial_metric = soln_sqrt_det_spatial_metric(coords, time)
    spatial_metric = soln_spatial_metric(coords, time)
    inv_spatial_metric = soln_inverse_spatial_metric(coords, time)

    return fluxes.tilde_b_flux(
        tilde_e,
        tilde_b,
        tilde_psi,
        tilde_phi,
        tilde_q,
        current.tilde_j(
            tilde_q,
            tilde_e,
            tilde_b,
            parallel_conductivity,
            lapse,
            sqrt_det_spatial_metric,
            spatial_metric,
        ),
        lapse,
        shift,
        sqrt_det_spatial_metric,
        spatial_metric,
        inv_spatial_metric,
    )


def data_flux_tilde_psi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    tilde_e = breakdown.TildeE(coords)
    tilde_b = breakdown.TildeB(coords)
    tilde_psi = breakdown.TildePsi(coords)
    tilde_phi = breakdown.TildePhi(coords)
    tilde_q = breakdown.TildeQ(coords)
    lapse = soln_lapse(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    shift = soln_shift(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    sqrt_det_spatial_metric = soln_sqrt_det_spatial_metric(coords, time)
    spatial_metric = soln_spatial_metric(coords, time)
    inv_spatial_metric = soln_inverse_spatial_metric(coords, time)

    return fluxes.tilde_psi_flux(
        tilde_e,
        tilde_b,
        tilde_psi,
        tilde_phi,
        tilde_q,
        current.tilde_j(
            tilde_q,
            tilde_e,
            tilde_b,
            parallel_conductivity,
            lapse,
            sqrt_det_spatial_metric,
            spatial_metric,
        ),
        lapse,
        shift,
        sqrt_det_spatial_metric,
        spatial_metric,
        inv_spatial_metric,
    )


def data_flux_tilde_phi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    tilde_e = breakdown.TildeE(coords)
    tilde_b = breakdown.TildeB(coords)
    tilde_psi = breakdown.TildePsi(coords)
    tilde_phi = breakdown.TildePhi(coords)
    tilde_q = breakdown.TildeQ(coords)
    lapse = soln_lapse(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    shift = soln_shift(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    sqrt_det_spatial_metric = soln_sqrt_det_spatial_metric(coords, time)
    spatial_metric = soln_spatial_metric(coords, time)
    inv_spatial_metric = soln_inverse_spatial_metric(coords, time)

    return fluxes.tilde_phi_flux(
        tilde_e,
        tilde_b,
        tilde_psi,
        tilde_phi,
        tilde_q,
        current.tilde_j(
            tilde_q,
            tilde_e,
            tilde_b,
            parallel_conductivity,
            lapse,
            sqrt_det_spatial_metric,
            spatial_metric,
        ),
        lapse,
        shift,
        sqrt_det_spatial_metric,
        spatial_metric,
        inv_spatial_metric,
    )


def data_flux_tilde_q(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    coords,
    time,
    dim,
    parallel_conductivity,
):
    tilde_e = breakdown.TildeE(coords)
    tilde_b = breakdown.TildeB(coords)
    tilde_psi = breakdown.TildePsi(coords)
    tilde_phi = breakdown.TildePhi(coords)
    tilde_q = breakdown.TildeQ(coords)
    lapse = soln_lapse(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    shift = soln_shift(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        coords,
        time,
        dim,
        parallel_conductivity,
    )
    sqrt_det_spatial_metric = soln_sqrt_det_spatial_metric(coords, time)
    spatial_metric = soln_spatial_metric(coords, time)
    inv_spatial_metric = soln_inverse_spatial_metric(coords, time)

    return fluxes.tilde_q_flux(
        tilde_e,
        tilde_b,
        tilde_psi,
        tilde_phi,
        tilde_q,
        current.tilde_j(
            tilde_q,
            tilde_e,
            tilde_b,
            parallel_conductivity,
            lapse,
            sqrt_det_spatial_metric,
            spatial_metric,
        ),
        lapse,
        shift,
        sqrt_det_spatial_metric,
        spatial_metric,
        inv_spatial_metric,
    )
