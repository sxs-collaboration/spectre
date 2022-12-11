// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/HydroFreeOutflow.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DgSubcell/SliceTensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::BoundaryConditions {
HydroFreeOutflow::HydroFreeOutflow(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
HydroFreeOutflow::get_clone() const {
  return std::make_unique<HydroFreeOutflow>(*this);
}

void HydroFreeOutflow::pup(PUP::er& p) { BoundaryCondition::pup(p); }

// NOLINTNEXTLINE
PUP::able::PUP_ID HydroFreeOutflow::my_PUP_ID = 0;

std::optional<std::string> HydroFreeOutflow::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_ye_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,

    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        inv_spatial_metric,

    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*face_mesh_velocity*/,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_vector,

    const Scalar<DataVector>& interior_rest_mass_density,
    const Scalar<DataVector>& interior_electron_fraction,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_lorentz_factor,
    const Scalar<DataVector>& interior_pressure,
    const Scalar<DataVector>& interior_specific_enthalpy,

    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
    const Scalar<DataVector>& interior_lapse,
    const tnsr::II<DataVector, 3, Frame::Inertial>&
        interior_inv_spatial_metric) {
  const size_t number_of_grid_points = get(interior_rest_mass_density).size();

  // temp buffer to store
  //  * n_i v^i
  //  * v_{ghost}^i
  //  * divergence cleaning field on ghost zone
  //  * spatial metric
  //  * sqrt determinant of spatial metric
  Variables<tmpl::list<::Tags::TempScalar<0>,
                       hydro::Tags::SpatialVelocity<DataVector, 3>,
                       ::Tags::TempScalar<1>, gr::Tags::SpatialMetric<3>,
                       gr::Tags::SqrtDetSpatialMetric<>>>
      temp_buffer{number_of_grid_points};
  auto& normal_dot_interior_spatial_velocity =
      get<::Tags::TempScalar<0>>(temp_buffer);
  auto& exterior_spatial_velocity =
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(temp_buffer);
  auto& exterior_divergence_cleaning_field =
      get<::Tags::TempScalar<1>>(temp_buffer);
  auto& interior_spatial_metric = get<gr::Tags::SpatialMetric<3>>(temp_buffer);
  auto& interior_sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<>>(temp_buffer);

  get(*lapse) = get(interior_lapse);
  for (size_t i = 0; i < 3; ++i) {
    (*shift).get(i) = interior_shift.get(i);
    for (size_t j = 0; j < 3; ++j) {
      (*inv_spatial_metric).get(i, j) = interior_inv_spatial_metric.get(i, j);
    }
  }

  // spatial metric and sqrt determinant of spatial metric can be retrived from
  // Databox but only as gridless_tags with whole volume data (unlike all the
  // other arguments which are face tensors). Rather than doing expensive tensor
  // slicing operations on those, we just compute those two quantities from
  // inverse spatial metric as below.
  determinant_and_inverse(make_not_null(&interior_sqrt_det_spatial_metric),
                          make_not_null(&interior_spatial_metric),
                          interior_inv_spatial_metric);
  get(interior_sqrt_det_spatial_metric) =
      1.0 / sqrt(get(interior_sqrt_det_spatial_metric));

  // copy-paste interior spatial velocity to exterior spatial velocity, but
  // kill ingoing normal component to zero
  dot_product(make_not_null(&normal_dot_interior_spatial_velocity),
              outward_directed_normal_covector, interior_spatial_velocity);
  for (size_t i = 0; i < number_of_grid_points; ++i) {
    if (get(normal_dot_interior_spatial_velocity)[i] >= 0.0) {
      for (size_t spatial_index = 0; spatial_index < 3; ++spatial_index) {
        exterior_spatial_velocity.get(spatial_index)[i] =
            interior_spatial_velocity.get(spatial_index)[i];
      }
    } else {
      for (size_t spatial_index = 0; spatial_index < 3; ++spatial_index) {
        exterior_spatial_velocity.get(spatial_index)[i] =
            interior_spatial_velocity.get(spatial_index)[i] -
            get(normal_dot_interior_spatial_velocity)[i] *
                outward_directed_normal_vector.get(spatial_index)[i];
      }
    }
  }

  get(exterior_divergence_cleaning_field) = 0.0;

  ConservativeFromPrimitive::apply(
      tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi,
      interior_rest_mass_density, interior_electron_fraction,
      interior_specific_internal_energy, interior_specific_enthalpy,
      interior_pressure, exterior_spatial_velocity, interior_lorentz_factor,
      interior_magnetic_field, interior_sqrt_det_spatial_metric,
      interior_spatial_metric, exterior_divergence_cleaning_field);

  ComputeFluxes::apply(tilde_d_flux, tilde_ye_flux, tilde_tau_flux,
                       tilde_s_flux, tilde_b_flux, tilde_phi_flux, *tilde_d,
                       *tilde_ye, *tilde_tau, *tilde_s, *tilde_b, *tilde_phi,
                       *lapse, *shift, interior_sqrt_det_spatial_metric,
                       interior_spatial_metric, *inv_spatial_metric,
                       interior_pressure, exterior_spatial_velocity,
                       interior_lorentz_factor, interior_magnetic_field);

  return {};
}

void HydroFreeOutflow::fd_ghost(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        lorentz_factor_times_spatial_velocity,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,

    const Direction<3>& direction,

    // fd_interior_temporary_tags
    const Mesh<3>& subcell_mesh,

    // fd_interior_primitive_variables_tags
    const Scalar<DataVector>& interior_rest_mass_density,
    const Scalar<DataVector>& interior_electron_fraction,
    const Scalar<DataVector>& interior_pressure,
    const Scalar<DataVector>& interior_lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,

    // fd_gridless_tags
    const fd::Reconstructor& reconstructor) {
  fd_ghost_impl(rest_mass_density, electron_fraction, pressure,
                lorentz_factor_times_spatial_velocity, magnetic_field,
                divergence_cleaning_field,

                direction,

                // fd_interior_temporary_tags
                subcell_mesh,

                // fd_interior_primitive_variables_tags
                interior_rest_mass_density, interior_electron_fraction,
                interior_pressure, interior_lorentz_factor,
                interior_spatial_velocity, interior_magnetic_field,

                reconstructor.ghost_zone_size());
}

void HydroFreeOutflow::fd_ghost_impl(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        lorentz_factor_times_spatial_velocity,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,

    const Direction<3>& direction,

    // fd_interior_temporary_tags
    const Mesh<3>& subcell_mesh,

    // fd_interior_primitive_variables_tags
    const Scalar<DataVector>& interior_rest_mass_density,
    const Scalar<DataVector>& interior_electron_fraction,
    const Scalar<DataVector>& interior_pressure,
    const Scalar<DataVector>& interior_lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,

    const size_t ghost_zone_size) {
  const size_t dim_direction{direction.dimension()};

  const auto subcell_extents{subcell_mesh.extents()};
  const size_t num_face_pts{
      subcell_extents.slice_away(dim_direction).product()};

  using prim_tags_without_cleaning_field =
      tmpl::list<RestMassDensity, ElectronFraction, Pressure,
                 LorentzFactorTimesSpatialVelocity, MagneticField>;

  // Create a single large DV to reduce the number of Variables allocations
  const size_t buffer_size_per_grid_pts =
      (*rest_mass_density).size() + (*electron_fraction).size() +
      (*pressure).size() + (*lorentz_factor_times_spatial_velocity).size() +
      (*magnetic_field).size();
  DataVector buffer_for_vars{
      num_face_pts * ((1 + ghost_zone_size) * buffer_size_per_grid_pts), 0.0};

  // Assign two Variables object to the buffer
  Variables<prim_tags_without_cleaning_field> outermost_prim_vars{
      buffer_for_vars.data(), num_face_pts * buffer_size_per_grid_pts};
  Variables<prim_tags_without_cleaning_field> ghost_prim_vars{
      outermost_prim_vars.data() + outermost_prim_vars.size(),
      num_face_pts * buffer_size_per_grid_pts * ghost_zone_size};

  // Slice values on the outermost grid points and store them to
  // `outermost_prim_vars`
  {
    auto get_boundary_val = [&direction, &subcell_extents](auto volume_tensor) {
      return evolution::dg::subcell::slice_tensor_for_subcell(
          volume_tensor, subcell_extents, 1, direction);
    };

    get<RestMassDensity>(outermost_prim_vars) =
        get_boundary_val(interior_rest_mass_density);
    get<ElectronFraction>(outermost_prim_vars) =
        get_boundary_val(interior_electron_fraction);
    get<Pressure>(outermost_prim_vars) = get_boundary_val(interior_pressure);

    // Kill ingoing components of spatial velocity and compute Wv^i
    //
    // Note : Here we require the grid to be Cartesian, therefore we will need
    // to change the implementation below once subcell supports curved mesh.
    const auto normal_spatial_velocity_at_boundary =
        get_boundary_val(interior_spatial_velocity).get(dim_direction);
    for (size_t i = 0; i < 3; ++i) {
      if (i == dim_direction) {
        if (direction.sign() > 0.0) {
          get<LorentzFactorTimesSpatialVelocity>(outermost_prim_vars).get(i) =
              get(get_boundary_val(interior_lorentz_factor)) *
              max(normal_spatial_velocity_at_boundary,
                  normal_spatial_velocity_at_boundary * 0.0);
        } else {
          get<LorentzFactorTimesSpatialVelocity>(outermost_prim_vars).get(i) =
              get(get_boundary_val(interior_lorentz_factor)) *
              min(normal_spatial_velocity_at_boundary,
                  normal_spatial_velocity_at_boundary * 0.0);
        }
      } else {
        get<LorentzFactorTimesSpatialVelocity>(outermost_prim_vars).get(i) =
            get(get_boundary_val(interior_lorentz_factor)) *
            get_boundary_val(interior_spatial_velocity).get(i);
      }

      get<MagneticField>(outermost_prim_vars).get(i) =
          get_boundary_val(interior_magnetic_field).get(i);
    }
  }

  // Now copy `outermost_prim_vars` into each slices of `ghost_prim_vars`.
  Index<3> ghost_data_extents = subcell_extents;
  ghost_data_extents[dim_direction] = ghost_zone_size;

  for (size_t i_ghost = 0; i_ghost < ghost_zone_size; ++i_ghost) {
    add_slice_to_data(make_not_null(&ghost_prim_vars), outermost_prim_vars,
                      ghost_data_extents, dim_direction, i_ghost);
  }

  *rest_mass_density = get<RestMassDensity>(ghost_prim_vars);
  *electron_fraction = get<ElectronFraction>(ghost_prim_vars);
  *pressure = get<Pressure>(ghost_prim_vars);
  *lorentz_factor_times_spatial_velocity =
      get<LorentzFactorTimesSpatialVelocity>(ghost_prim_vars);
  *magnetic_field = get<MagneticField>(ghost_prim_vars);

  // divergence cleaning scalar field is just set to zero in the ghost zone.
  get(*divergence_cleaning_field) = 0.0;
}
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
