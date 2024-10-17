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
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DgSubcell/SliceTensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
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
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        spatial_velocity_one_form,
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        spatial_velocity,
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
    const Scalar<DataVector>& interior_temperature,

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
  Variables<tmpl::list<
      ::Tags::TempScalar<0>, hydro::Tags::SpatialVelocity<DataVector, 3>,
      ::Tags::TempScalar<1>, gr::Tags::SpatialMetric<DataVector, 3>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>>>
      temp_buffer{number_of_grid_points};
  auto& normal_dot_interior_spatial_velocity =
      get<::Tags::TempScalar<0>>(temp_buffer);
  *spatial_velocity =
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(temp_buffer);
  auto& exterior_divergence_cleaning_field =
      get<::Tags::TempScalar<1>>(temp_buffer);
  auto& interior_spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(temp_buffer);
  auto& interior_sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(temp_buffer);

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
        spatial_velocity->get(spatial_index)[i] =
            interior_spatial_velocity.get(spatial_index)[i];
      }
    } else {
      for (size_t spatial_index = 0; spatial_index < 3; ++spatial_index) {
        spatial_velocity->get(spatial_index)[i] =
            interior_spatial_velocity.get(spatial_index)[i] -
            get(normal_dot_interior_spatial_velocity)[i] *
                outward_directed_normal_vector.get(spatial_index)[i];
      }
    }
  }

  get(exterior_divergence_cleaning_field) = 0.0;

  *rest_mass_density = interior_rest_mass_density;
  *electron_fraction = interior_electron_fraction;
  *temperature = interior_temperature;
  tenex::evaluate<ti::i>(
      spatial_velocity_one_form,
      (*spatial_velocity)(ti::J) * (interior_spatial_metric(ti::i, ti::j)));

  ConservativeFromPrimitive::apply(
      tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi,
      interior_rest_mass_density, interior_electron_fraction,
      interior_specific_internal_energy, interior_pressure, *spatial_velocity,
      interior_lorentz_factor, interior_magnetic_field,
      interior_sqrt_det_spatial_metric, interior_spatial_metric,
      exterior_divergence_cleaning_field);

  ComputeFluxes::apply(
      tilde_d_flux, tilde_ye_flux, tilde_tau_flux, tilde_s_flux, tilde_b_flux,
      tilde_phi_flux, *tilde_d, *tilde_ye, *tilde_tau, *tilde_s, *tilde_b,
      *tilde_phi, *lapse, *shift, interior_sqrt_det_spatial_metric,
      interior_spatial_metric, *inv_spatial_metric, interior_pressure,
      *spatial_velocity, interior_lorentz_factor, interior_magnetic_field);

  return {};
}

namespace {
template <typename T>
using Flux = ::Tags::Flux<T, tmpl::size_t<3>, Frame::Inertial>;
}  // namespace

void HydroFreeOutflow::fd_ghost(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        lorentz_factor_times_spatial_velocity,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,

    const gsl::not_null<std::optional<Variables<db::wrap_tags_in<
        ::Tags::Flux, typename grmhd::ValenciaDivClean::System::flux_variables,
        tmpl::size_t<3>, Frame::Inertial>>>*>
        cell_centered_ghost_fluxes,

    const Direction<3>& direction,

    // fd_interior_temporary_tags
    const Mesh<3>& subcell_mesh,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
    const Scalar<DataVector>& interior_lapse,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_spatial_metric,

    // fd_interior_primitive_variables_tags
    const Scalar<DataVector>& interior_rest_mass_density,
    const Scalar<DataVector>& interior_electron_fraction,
    const Scalar<DataVector>& interior_temperature,
    const Scalar<DataVector>& interior_pressure,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const Scalar<DataVector>& interior_lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,

    // fd_gridless_tags
    const fd::Reconstructor& reconstructor) {
  Variables<tmpl::push_back<typename System::variables_tag::tags_list,
                            SpatialVelocity, LorentzFactor, Pressure,
                            SpecificInternalEnergy, SqrtDetSpatialMetric,
                            SpatialMetric, InvSpatialMetric, Lapse, Shift>>
      temp_vars{get(*rest_mass_density).size()};
  fd_ghost_impl(rest_mass_density, electron_fraction, temperature,
                make_not_null(&get<Pressure>(temp_vars)),
                make_not_null(&get<SpecificInternalEnergy>(temp_vars)),
                lorentz_factor_times_spatial_velocity,
                make_not_null(&get<SpatialVelocity>(temp_vars)),
                make_not_null(&get<LorentzFactor>(temp_vars)), magnetic_field,
                divergence_cleaning_field,
                make_not_null(&get<SpatialMetric>(temp_vars)),
                make_not_null(&get<InvSpatialMetric>(temp_vars)),
                make_not_null(&get<SqrtDetSpatialMetric>(temp_vars)),
                make_not_null(&get<Lapse>(temp_vars)),
                make_not_null(&get<Shift>(temp_vars)),

                direction,

                // fd_interior_temporary_tags
                subcell_mesh,

                // fd_interior_primitive_variables_tags
                interior_rest_mass_density, interior_electron_fraction,
                interior_temperature, interior_pressure,
                interior_specific_internal_energy, interior_lorentz_factor,
                interior_spatial_velocity, interior_magnetic_field,
                interior_spatial_metric, interior_lapse, interior_shift,

                reconstructor.ghost_zone_size(),

                cell_centered_ghost_fluxes->has_value());
  if (cell_centered_ghost_fluxes->has_value()) {
    ConservativeFromPrimitive::apply(
        make_not_null(&get<Tags::TildeD>(temp_vars)),
        make_not_null(&get<Tags::TildeYe>(temp_vars)),
        make_not_null(&get<Tags::TildeTau>(temp_vars)),
        make_not_null(&get<Tags::TildeS<>>(temp_vars)),
        make_not_null(&get<Tags::TildeB<>>(temp_vars)),
        make_not_null(&get<Tags::TildePhi>(temp_vars)),

        // Note: Only the spatial velocity changes.
        *rest_mass_density, *electron_fraction,
        get<SpecificInternalEnergy>(temp_vars), get<Pressure>(temp_vars),
        get<SpatialVelocity>(temp_vars), get<LorentzFactor>(temp_vars),
        *magnetic_field,

        get<SqrtDetSpatialMetric>(temp_vars), get<SpatialMetric>(temp_vars),
        *divergence_cleaning_field);

    ComputeFluxes::apply(
        make_not_null(
            &get<Flux<Tags::TildeD>>(cell_centered_ghost_fluxes->value())),
        make_not_null(
            &get<Flux<Tags::TildeYe>>(cell_centered_ghost_fluxes->value())),
        make_not_null(
            &get<Flux<Tags::TildeTau>>(cell_centered_ghost_fluxes->value())),
        make_not_null(
            &get<Flux<Tags::TildeS<>>>(cell_centered_ghost_fluxes->value())),
        make_not_null(
            &get<Flux<Tags::TildeB<>>>(cell_centered_ghost_fluxes->value())),
        make_not_null(
            &get<Flux<Tags::TildePhi>>(cell_centered_ghost_fluxes->value())),

        get<Tags::TildeD>(temp_vars), get<Tags::TildeYe>(temp_vars),
        get<Tags::TildeTau>(temp_vars), get<Tags::TildeS<>>(temp_vars),
        get<Tags::TildeB<>>(temp_vars), get<Tags::TildePhi>(temp_vars),

        get<Lapse>(temp_vars), get<Shift>(temp_vars),
        get<SqrtDetSpatialMetric>(temp_vars), get<SpatialMetric>(temp_vars),
        get<InvSpatialMetric>(temp_vars), get<Pressure>(temp_vars),
        get<SpatialVelocity>(temp_vars), get<LorentzFactor>(temp_vars),
        *magnetic_field);
  }
}

void HydroFreeOutflow::fd_ghost_impl(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        lorentz_factor_times_spatial_velocity,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        spatial_metric,
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        inv_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> sqrt_det_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,

    const Direction<3>& direction,

    // fd_interior_temporary_tags
    const Mesh<3>& subcell_mesh,

    // fd_interior_primitive_variables_tags
    const Scalar<DataVector>& interior_rest_mass_density,
    const Scalar<DataVector>& interior_electron_fraction,
    const Scalar<DataVector>& interior_temperature,
    const Scalar<DataVector>& interior_pressure,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const Scalar<DataVector>& interior_lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_spatial_metric,
    const Scalar<DataVector>& interior_lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,

    const size_t ghost_zone_size, const bool need_tags_for_fluxes) {
  const size_t dim_direction{direction.dimension()};

  const auto subcell_extents{subcell_mesh.extents()};
  const size_t num_face_pts{
      subcell_extents.slice_away(dim_direction).product()};

  using prim_tags_without_cleaning_field =
      tmpl::list<RestMassDensity, ElectronFraction, Temperature,
                 LorentzFactorTimesSpatialVelocity, MagneticField>;

  // Create a single large DV to reduce the number of Variables allocations
  using fluxes_tags =
      tmpl::list<Pressure, SpecificInternalEnergy, SpatialMetric, Lapse, Shift>;
  const size_t buffer_size_for_fluxes =
      need_tags_for_fluxes
          ? Variables<fluxes_tags>::number_of_independent_components
          : 0;
  const size_t buffer_size_per_grid_pts = Variables<
      prim_tags_without_cleaning_field>::number_of_independent_components;
  DataVector buffer_for_vars{
      num_face_pts * ((1 + ghost_zone_size) *
                      (buffer_size_per_grid_pts + buffer_size_for_fluxes)),
      0.0};

  // Assign two Variables object to the buffer
  Variables<prim_tags_without_cleaning_field> outermost_prim_vars{
      buffer_for_vars.data(), num_face_pts * buffer_size_per_grid_pts};
  Variables<prim_tags_without_cleaning_field> ghost_prim_vars{
      outermost_prim_vars.data() + outermost_prim_vars.size(),
      num_face_pts * buffer_size_per_grid_pts * ghost_zone_size};

  // Slice values on the outermost grid points and store them to
  // `outermost_prim_vars`
  auto get_boundary_val = [&direction, &subcell_extents](auto volume_tensor) {
    return evolution::dg::subcell::slice_tensor_for_subcell(
        volume_tensor, subcell_extents, 1, direction, {});
  };

  get<RestMassDensity>(outermost_prim_vars) =
      get_boundary_val(interior_rest_mass_density);
  get<ElectronFraction>(outermost_prim_vars) =
      get_boundary_val(interior_electron_fraction);
  get<Temperature>(outermost_prim_vars) =
      get_boundary_val(interior_temperature);

  {
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
  *temperature = get<Temperature>(ghost_prim_vars);
  *lorentz_factor_times_spatial_velocity =
      get<LorentzFactorTimesSpatialVelocity>(ghost_prim_vars);
  *magnetic_field = get<MagneticField>(ghost_prim_vars);

  // divergence cleaning scalar field is just set to zero in the ghost zone.
  get(*divergence_cleaning_field) = 0.0;

  if (need_tags_for_fluxes) {
    Variables<fluxes_tags> outermost_fluxes_vars{
        std::next(ghost_prim_vars.data(),
                  static_cast<std::ptrdiff_t>(ghost_prim_vars.size())),
        num_face_pts * buffer_size_for_fluxes};
    Variables<fluxes_tags> ghost_fluxes_vars{
        std::next(outermost_fluxes_vars.data(),
                  static_cast<std::ptrdiff_t>(outermost_fluxes_vars.size())),
        num_face_pts * buffer_size_for_fluxes * ghost_zone_size};

    get<Pressure>(outermost_fluxes_vars) = get_boundary_val(interior_pressure);
    get<SpecificInternalEnergy>(outermost_fluxes_vars) =
        get_boundary_val(interior_specific_internal_energy);
    get<SpatialMetric>(outermost_fluxes_vars) =
        get_boundary_val(interior_spatial_metric);
    get<Lapse>(outermost_fluxes_vars) = get_boundary_val(interior_lapse);
    get<Shift>(outermost_fluxes_vars) = get_boundary_val(interior_shift);

    for (size_t i_ghost = 0; i_ghost < ghost_zone_size; ++i_ghost) {
      add_slice_to_data(make_not_null(&ghost_fluxes_vars),
                        outermost_fluxes_vars, ghost_data_extents,
                        dim_direction, i_ghost);
    }
    // Need pressure for high-order finite difference
    *pressure = get<Pressure>(ghost_fluxes_vars);
    *specific_internal_energy = get<SpecificInternalEnergy>(ghost_fluxes_vars);
    *spatial_metric = get<SpatialMetric>(ghost_fluxes_vars);
    *lapse = get<Lapse>(ghost_fluxes_vars);
    *shift = get<Shift>(ghost_fluxes_vars);

    determinant_and_inverse(sqrt_det_spatial_metric, inv_spatial_metric,
                            *spatial_metric);
    get(*sqrt_det_spatial_metric) = sqrt(get(*sqrt_det_spatial_metric));
    tenex::evaluate(
        lorentz_factor,
        sqrt(1.0 + (*spatial_metric)(ti::i, ti::j) *
                       (*lorentz_factor_times_spatial_velocity)(ti::I) *
                       (*lorentz_factor_times_spatial_velocity)(ti::J)));
    tenex::evaluate<ti::I>(
        spatial_velocity,
        (*lorentz_factor_times_spatial_velocity)(ti::I) / (*lorentz_factor)());
  }
}
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
