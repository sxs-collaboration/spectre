# Distributed under the MIT License.
# See LICENSE.txt for details.

import Evolution.Systems.GrMhd.ValenciaDivClean.Fluxes as fluxes
import Evolution.Systems.GrMhd.ValenciaDivClean.TestFunctions as cons
import numpy as np


def _exterior_spatial_velocity(
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_spatial_velocity,
):
    dot_interior_spatial_velocity = np.dot(
        outward_directed_normal_covector, interior_spatial_velocity
    )
    return np.where(
        dot_interior_spatial_velocity > 0.0,
        interior_spatial_velocity,
        interior_spatial_velocity
        - outward_directed_normal_vector * dot_interior_spatial_velocity,
    )


def _spatial_metric(inv_spatial_metric):
    return np.linalg.inv(inv_spatial_metric)


def error(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    return None


def tilde_d(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    exterior_spatial_velocity = _exterior_spatial_velocity(
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_spatial_velocity,
    )
    spatial_metric = _spatial_metric(inv_spatial_metric)
    sqrt_det_spatial_metric = np.sqrt(np.linalg.det(spatial_metric))

    return cons.tilde_d(
        interior_rest_mass_density,
        interior_electron_fraction,
        interior_specific_internal_energy,
        interior_pressure,
        exterior_spatial_velocity,
        interior_lorentz_factor,
        interior_magnetic_field,
        sqrt_det_spatial_metric,
        spatial_metric,
        0.0,
    )


def tilde_ye(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    exterior_spatial_velocity = _exterior_spatial_velocity(
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_spatial_velocity,
    )
    spatial_metric = _spatial_metric(inv_spatial_metric)
    sqrt_det_spatial_metric = np.sqrt(np.linalg.det(spatial_metric))

    return cons.tilde_ye(
        interior_rest_mass_density,
        interior_electron_fraction,
        interior_specific_internal_energy,
        interior_pressure,
        exterior_spatial_velocity,
        interior_lorentz_factor,
        interior_magnetic_field,
        sqrt_det_spatial_metric,
        spatial_metric,
        0.0,
    )


def tilde_tau(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    exterior_spatial_velocity = _exterior_spatial_velocity(
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_spatial_velocity,
    )
    spatial_metric = _spatial_metric(inv_spatial_metric)
    sqrt_det_spatial_metric = np.sqrt(np.linalg.det(spatial_metric))

    return cons.tilde_tau(
        interior_rest_mass_density,
        interior_electron_fraction,
        interior_specific_internal_energy,
        interior_pressure,
        exterior_spatial_velocity,
        interior_lorentz_factor,
        interior_magnetic_field,
        sqrt_det_spatial_metric,
        spatial_metric,
        0.0,
    )


def tilde_s(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    exterior_spatial_velocity = _exterior_spatial_velocity(
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_spatial_velocity,
    )
    spatial_metric = _spatial_metric(inv_spatial_metric)
    sqrt_det_spatial_metric = np.sqrt(np.linalg.det(spatial_metric))

    return cons.tilde_s(
        interior_rest_mass_density,
        interior_electron_fraction,
        interior_specific_internal_energy,
        interior_pressure,
        exterior_spatial_velocity,
        interior_lorentz_factor,
        interior_magnetic_field,
        sqrt_det_spatial_metric,
        spatial_metric,
        0.0,
    )


def tilde_b(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    exterior_spatial_velocity = _exterior_spatial_velocity(
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_spatial_velocity,
    )
    spatial_metric = _spatial_metric(inv_spatial_metric)
    sqrt_det_spatial_metric = np.sqrt(np.linalg.det(spatial_metric))

    return cons.tilde_b(
        interior_rest_mass_density,
        interior_electron_fraction,
        interior_specific_internal_energy,
        interior_pressure,
        exterior_spatial_velocity,
        interior_lorentz_factor,
        interior_magnetic_field,
        sqrt_det_spatial_metric,
        spatial_metric,
        0.0,
    )


def tilde_phi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    exterior_spatial_velocity = _exterior_spatial_velocity(
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_spatial_velocity,
    )
    spatial_metric = _spatial_metric(inv_spatial_metric)
    sqrt_det_spatial_metric = np.sqrt(np.linalg.det(spatial_metric))

    return cons.tilde_phi(
        interior_rest_mass_density,
        interior_electron_fraction,
        interior_specific_internal_energy,
        interior_pressure,
        exterior_spatial_velocity,
        interior_lorentz_factor,
        interior_magnetic_field,
        sqrt_det_spatial_metric,
        spatial_metric,
        0.0,
    )


def _return_cons_vars(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    return {
        "tilde_d": tilde_d(
            face_mesh_velocity,
            outward_directed_normal_covector,
            outward_directed_normal_vector,
            interior_rest_mass_density,
            interior_electron_fraction,
            interior_specific_internal_energy,
            interior_spatial_velocity,
            interior_magnetic_field,
            interior_lorentz_factor,
            interior_pressure,
            interior_temperature,
            shift,
            lapse,
            inv_spatial_metric,
        ),
        "tilde_ye": tilde_ye(
            face_mesh_velocity,
            outward_directed_normal_covector,
            outward_directed_normal_vector,
            interior_rest_mass_density,
            interior_electron_fraction,
            interior_specific_internal_energy,
            interior_spatial_velocity,
            interior_magnetic_field,
            interior_lorentz_factor,
            interior_pressure,
            interior_temperature,
            shift,
            lapse,
            inv_spatial_metric,
        ),
        "tilde_tau": tilde_tau(
            face_mesh_velocity,
            outward_directed_normal_covector,
            outward_directed_normal_vector,
            interior_rest_mass_density,
            interior_electron_fraction,
            interior_specific_internal_energy,
            interior_spatial_velocity,
            interior_magnetic_field,
            interior_lorentz_factor,
            interior_pressure,
            interior_temperature,
            shift,
            lapse,
            inv_spatial_metric,
        ),
        "tilde_s": tilde_s(
            face_mesh_velocity,
            outward_directed_normal_covector,
            outward_directed_normal_vector,
            interior_rest_mass_density,
            interior_electron_fraction,
            interior_specific_internal_energy,
            interior_spatial_velocity,
            interior_magnetic_field,
            interior_lorentz_factor,
            interior_pressure,
            interior_temperature,
            shift,
            lapse,
            inv_spatial_metric,
        ),
        "tilde_b": tilde_b(
            face_mesh_velocity,
            outward_directed_normal_covector,
            outward_directed_normal_vector,
            interior_rest_mass_density,
            interior_electron_fraction,
            interior_specific_internal_energy,
            interior_spatial_velocity,
            interior_magnetic_field,
            interior_lorentz_factor,
            interior_pressure,
            interior_temperature,
            shift,
            lapse,
            inv_spatial_metric,
        ),
        "tilde_phi": tilde_phi(
            face_mesh_velocity,
            outward_directed_normal_covector,
            outward_directed_normal_vector,
            interior_rest_mass_density,
            interior_electron_fraction,
            interior_specific_internal_energy,
            interior_spatial_velocity,
            interior_magnetic_field,
            interior_lorentz_factor,
            interior_pressure,
            interior_temperature,
            shift,
            lapse,
            inv_spatial_metric,
        ),
    }


def flux_tilde_d(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    exterior_spatial_velocity = _exterior_spatial_velocity(
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_spatial_velocity,
    )
    spatial_metric = _spatial_metric(inv_spatial_metric)
    sqrt_det_spatial_metric = np.sqrt(np.linalg.det(spatial_metric))

    cons_vars = _return_cons_vars(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_rest_mass_density,
        interior_electron_fraction,
        interior_specific_internal_energy,
        interior_spatial_velocity,
        interior_magnetic_field,
        interior_lorentz_factor,
        interior_pressure,
        interior_temperature,
        shift,
        lapse,
        inv_spatial_metric,
    )

    return fluxes.tilde_d_flux(
        cons_vars["tilde_d"],
        cons_vars["tilde_ye"],
        cons_vars["tilde_tau"],
        cons_vars["tilde_s"],
        cons_vars["tilde_b"],
        cons_vars["tilde_phi"],
        lapse,
        shift,
        sqrt_det_spatial_metric,
        spatial_metric,
        inv_spatial_metric,
        interior_pressure,
        exterior_spatial_velocity,
        interior_lorentz_factor,
        interior_magnetic_field,
    )


def flux_tilde_ye(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    exterior_spatial_velocity = _exterior_spatial_velocity(
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_spatial_velocity,
    )
    spatial_metric = _spatial_metric(inv_spatial_metric)
    sqrt_det_spatial_metric = np.sqrt(np.linalg.det(spatial_metric))

    cons_vars = _return_cons_vars(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_rest_mass_density,
        interior_electron_fraction,
        interior_specific_internal_energy,
        interior_spatial_velocity,
        interior_magnetic_field,
        interior_lorentz_factor,
        interior_pressure,
        interior_temperature,
        shift,
        lapse,
        inv_spatial_metric,
    )

    return fluxes.tilde_ye_flux(
        cons_vars["tilde_d"],
        cons_vars["tilde_ye"],
        cons_vars["tilde_tau"],
        cons_vars["tilde_s"],
        cons_vars["tilde_b"],
        cons_vars["tilde_phi"],
        lapse,
        shift,
        sqrt_det_spatial_metric,
        spatial_metric,
        inv_spatial_metric,
        interior_pressure,
        exterior_spatial_velocity,
        interior_lorentz_factor,
        interior_magnetic_field,
    )


def flux_tilde_tau(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    exterior_spatial_velocity = _exterior_spatial_velocity(
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_spatial_velocity,
    )
    spatial_metric = _spatial_metric(inv_spatial_metric)
    sqrt_det_spatial_metric = np.sqrt(np.linalg.det(spatial_metric))

    cons_vars = _return_cons_vars(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_rest_mass_density,
        interior_electron_fraction,
        interior_specific_internal_energy,
        interior_spatial_velocity,
        interior_magnetic_field,
        interior_lorentz_factor,
        interior_pressure,
        interior_temperature,
        shift,
        lapse,
        inv_spatial_metric,
    )

    return fluxes.tilde_tau_flux(
        cons_vars["tilde_d"],
        cons_vars["tilde_ye"],
        cons_vars["tilde_tau"],
        cons_vars["tilde_s"],
        cons_vars["tilde_b"],
        cons_vars["tilde_phi"],
        lapse,
        shift,
        sqrt_det_spatial_metric,
        spatial_metric,
        inv_spatial_metric,
        interior_pressure,
        exterior_spatial_velocity,
        interior_lorentz_factor,
        interior_magnetic_field,
    )


def flux_tilde_s(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    exterior_spatial_velocity = _exterior_spatial_velocity(
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_spatial_velocity,
    )
    spatial_metric = _spatial_metric(inv_spatial_metric)
    sqrt_det_spatial_metric = np.sqrt(np.linalg.det(spatial_metric))

    cons_vars = _return_cons_vars(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_rest_mass_density,
        interior_electron_fraction,
        interior_specific_internal_energy,
        interior_spatial_velocity,
        interior_magnetic_field,
        interior_lorentz_factor,
        interior_pressure,
        interior_temperature,
        shift,
        lapse,
        inv_spatial_metric,
    )

    return fluxes.tilde_s_flux(
        cons_vars["tilde_d"],
        cons_vars["tilde_ye"],
        cons_vars["tilde_tau"],
        cons_vars["tilde_s"],
        cons_vars["tilde_b"],
        cons_vars["tilde_phi"],
        lapse,
        shift,
        sqrt_det_spatial_metric,
        spatial_metric,
        inv_spatial_metric,
        interior_pressure,
        exterior_spatial_velocity,
        interior_lorentz_factor,
        interior_magnetic_field,
    )


def flux_tilde_b(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    exterior_spatial_velocity = _exterior_spatial_velocity(
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_spatial_velocity,
    )
    spatial_metric = _spatial_metric(inv_spatial_metric)
    sqrt_det_spatial_metric = np.sqrt(np.linalg.det(spatial_metric))

    cons_vars = _return_cons_vars(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_rest_mass_density,
        interior_electron_fraction,
        interior_specific_internal_energy,
        interior_spatial_velocity,
        interior_magnetic_field,
        interior_lorentz_factor,
        interior_pressure,
        interior_temperature,
        shift,
        lapse,
        inv_spatial_metric,
    )

    return fluxes.tilde_b_flux(
        cons_vars["tilde_d"],
        cons_vars["tilde_ye"],
        cons_vars["tilde_tau"],
        cons_vars["tilde_s"],
        cons_vars["tilde_b"],
        cons_vars["tilde_phi"],
        lapse,
        shift,
        sqrt_det_spatial_metric,
        spatial_metric,
        inv_spatial_metric,
        interior_pressure,
        exterior_spatial_velocity,
        interior_lorentz_factor,
        interior_magnetic_field,
    )


def flux_tilde_phi(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    exterior_spatial_velocity = _exterior_spatial_velocity(
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_spatial_velocity,
    )
    spatial_metric = _spatial_metric(inv_spatial_metric)
    sqrt_det_spatial_metric = np.sqrt(np.linalg.det(spatial_metric))

    cons_vars = _return_cons_vars(
        face_mesh_velocity,
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_rest_mass_density,
        interior_electron_fraction,
        interior_specific_internal_energy,
        interior_spatial_velocity,
        interior_magnetic_field,
        interior_lorentz_factor,
        interior_pressure,
        interior_temperature,
        shift,
        lapse,
        inv_spatial_metric,
    )

    return fluxes.tilde_phi_flux(
        cons_vars["tilde_d"],
        cons_vars["tilde_ye"],
        cons_vars["tilde_tau"],
        cons_vars["tilde_s"],
        cons_vars["tilde_b"],
        cons_vars["tilde_phi"],
        lapse,
        shift,
        sqrt_det_spatial_metric,
        spatial_metric,
        inv_spatial_metric,
        interior_pressure,
        exterior_spatial_velocity,
        interior_lorentz_factor,
        interior_magnetic_field,
    )


def lapse(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    return lapse


def shift(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    return shift


def inv_spatial_metric(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    return inv_spatial_metric


def spatial_velocity(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    return _exterior_spatial_velocity(
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_spatial_velocity,
    )


def spatial_velocity_one_form(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    exterior_spatial_velocity = _exterior_spatial_velocity(
        outward_directed_normal_covector,
        outward_directed_normal_vector,
        interior_spatial_velocity,
    )
    spatial_metric = _spatial_metric(inv_spatial_metric)
    return np.einsum("i, ij", exterior_spatial_velocity, spatial_metric)


def rest_mass_density(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    return interior_rest_mass_density


def electron_fraction(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    return interior_electron_fraction


def temperature(
    face_mesh_velocity,
    outward_directed_normal_covector,
    outward_directed_normal_vector,
    interior_rest_mass_density,
    interior_electron_fraction,
    interior_specific_internal_energy,
    interior_spatial_velocity,
    interior_magnetic_field,
    interior_lorentz_factor,
    interior_pressure,
    interior_temperature,
    shift,
    lapse,
    inv_spatial_metric,
):
    return interior_temperature
