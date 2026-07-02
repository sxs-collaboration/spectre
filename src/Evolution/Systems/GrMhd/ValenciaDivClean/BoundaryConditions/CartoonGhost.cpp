// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/CartoonGhost.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Expressions/Evaluate.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ComputeFluxesFromPrimitives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::BoundaryConditions {
CartoonGhost::CartoonGhost(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
CartoonGhost::get_clone() const {
  return std::make_unique<CartoonGhost>(*this);
}

void CartoonGhost::pup(PUP::er& p) { BoundaryCondition::pup(p); }
// NOLINTNEXTLINE
PUP::able::PUP_ID CartoonGhost::my_PUP_ID = 0;

void CartoonGhost::fd_ghost(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        lorentz_factor_times_spatial_velocity,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,

    const gsl::not_null<std::optional<Variables<db::wrap_tags_in<
        Flux, typename grmhd::ValenciaDivClean::System::flux_variables>>>*>
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
    const Scalar<DataVector>& interior_divergence_cleaning_field,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,

    // fd_gridless_tags
    const fd::Reconstructor& reconstructor) {
  Variables<tmpl::push_back<
      tmpl::append<typename System::variables_tag::tags_list,
                   typename System::primitive_variables_tag::tags_list>,
      SqrtDetSpatialMetric, SpatialMetric, InvSpatialMetric, Lapse, Shift>>
      temp_vars{get(*rest_mass_density).size()};

  fd_ghost_impl(make_not_null(&get<RestMassDensity>(temp_vars)),
                make_not_null(&get<ElectronFraction>(temp_vars)),
                make_not_null(&get<Temperature>(temp_vars)),
                make_not_null(&get<Pressure>(temp_vars)),
                make_not_null(&get<SpecificInternalEnergy>(temp_vars)),
                lorentz_factor_times_spatial_velocity,
                make_not_null(&get<SpatialVelocity>(temp_vars)),
                make_not_null(&get<LorentzFactor>(temp_vars)),
                make_not_null(&get<MagneticField>(temp_vars)),
                make_not_null(&get<DivergenceCleaningField>(temp_vars)),
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
                interior_divergence_cleaning_field, interior_spatial_velocity,
                interior_magnetic_field, interior_spatial_metric,
                interior_lapse, interior_shift,

                reconstructor.ghost_zone_size(),

                cell_centered_ghost_fluxes->has_value());

  if (cell_centered_ghost_fluxes->has_value()) {
    compute_fluxes_from_primitives(
        make_not_null(&cell_centered_ghost_fluxes->value()), temp_vars);
  }

  *rest_mass_density = get<hydro::Tags::RestMassDensity<DataVector>>(temp_vars);
  *electron_fraction =
      get<hydro::Tags::ElectronFraction<DataVector>>(temp_vars);
  *temperature = get<hydro::Tags::Temperature<DataVector>>(temp_vars);
  *magnetic_field = get<hydro::Tags::MagneticField<DataVector, 3>>(temp_vars);
  *divergence_cleaning_field =
      get<hydro::Tags::DivergenceCleaningField<DataVector>>(temp_vars);
}

void CartoonGhost::fd_ghost_impl(
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
    const Scalar<DataVector>& interior_divergence_cleaning_field,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_spatial_metric,
    const Scalar<DataVector>& interior_lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,

    const size_t ghost_zone_size, const bool need_tags_for_fluxes) {
  const size_t dim_direction{direction.dimension()};
  ASSERT(dim_direction == 0,
         "Cartoon BC can only be applied in the x-direction, got "
             << dim_direction);

  const auto subcell_extents{subcell_mesh.extents()};
  const size_t num_face_pts{
      subcell_extents.slice_away(dim_direction).product()};

  using prim_tags = tmpl::list<RestMassDensity, ElectronFraction, Temperature,
                               LorentzFactorTimesSpatialVelocity, MagneticField,
                               DivergenceCleaningField>;

  // Create a single large DV to reduce the number of Variables allocations
  using fluxes_tags =
      tmpl::list<Pressure, SpecificInternalEnergy, SpatialMetric, Lapse, Shift>;
  const size_t buffer_size_for_fluxes =
      need_tags_for_fluxes
          ? Variables<fluxes_tags>::number_of_independent_components
          : 0;
  const size_t buffer_size_per_grid_pts =
      Variables<prim_tags>::number_of_independent_components;
  DataVector buffer_for_vars{
      num_face_pts * ((1 + ghost_zone_size) *
                      (buffer_size_per_grid_pts + buffer_size_for_fluxes)),
      0.0};

  // outermost_prim_vars is scratch space for one ghost layer at a time.
  Variables<prim_tags> outermost_prim_vars{
      buffer_for_vars.data(), num_face_pts * buffer_size_per_grid_pts};
  Variables<prim_tags> ghost_prim_vars{
      outermost_prim_vars.data() + outermost_prim_vars.size(),
      num_face_pts * buffer_size_per_grid_pts * ghost_zone_size};

  // The ghost data stencil is ordered deepest-first: ghost[j=0] is the cell
  // furthest from the element at xi = -(ghost_zone_size-0.5)*dxi, and
  // ghost[j=ghost_zone_size-1] is nearest at xi = -0.5*dxi. By parity,
  // ghost[j=k] mirrors interior cell ix = ghost_zone_size-1-k.
  Index<3> ghost_data_extents = subcell_extents;
  ghost_data_extents[dim_direction] = ghost_zone_size;

  for (size_t k = 0; k < ghost_zone_size; ++k) {
    const size_t interior_ix = ghost_zone_size - 1 - k;

    get<RestMassDensity>(outermost_prim_vars) =
        data_on_slice(interior_rest_mass_density, subcell_extents,
                      dim_direction, interior_ix);
    get<ElectronFraction>(outermost_prim_vars) =
        data_on_slice(interior_electron_fraction, subcell_extents,
                      dim_direction, interior_ix);
    get<Temperature>(outermost_prim_vars) = data_on_slice(
        interior_temperature, subcell_extents, dim_direction, interior_ix);
    get<DivergenceCleaningField>(outermost_prim_vars) =
        data_on_slice(interior_divergence_cleaning_field, subcell_extents,
                      dim_direction, interior_ix);

    {
      // reflect both the outgoing and ingoing normal components of
      // spatial velocity and the magnetic field.
      const auto sliced_lorentz_factor = data_on_slice(
          interior_lorentz_factor, subcell_extents, dim_direction, interior_ix);
      const auto sliced_spatial_velocity =
          data_on_slice(interior_spatial_velocity, subcell_extents,
                        dim_direction, interior_ix);
      const auto sliced_magnetic_field = data_on_slice(
          interior_magnetic_field, subcell_extents, dim_direction, interior_ix);
      for (size_t i = 0; i < 3; ++i) {
        if (i == dim_direction) {
          get<LorentzFactorTimesSpatialVelocity>(outermost_prim_vars).get(i) =
              -get(sliced_lorentz_factor) * sliced_spatial_velocity.get(i);
          get<MagneticField>(outermost_prim_vars).get(i) =
              -1.0 * sliced_magnetic_field.get(i);
        } else {
          get<LorentzFactorTimesSpatialVelocity>(outermost_prim_vars).get(i) =
              get(sliced_lorentz_factor) * sliced_spatial_velocity.get(i);
          get<MagneticField>(outermost_prim_vars).get(i) =
              sliced_magnetic_field.get(i);
        }
      }
    }

    add_slice_to_data(make_not_null(&ghost_prim_vars), outermost_prim_vars,
                      ghost_data_extents, dim_direction, k);
  }

  *rest_mass_density = get<RestMassDensity>(ghost_prim_vars);
  *electron_fraction = get<ElectronFraction>(ghost_prim_vars);
  *temperature = get<Temperature>(ghost_prim_vars);
  *lorentz_factor_times_spatial_velocity =
      get<LorentzFactorTimesSpatialVelocity>(ghost_prim_vars);
  *magnetic_field = get<MagneticField>(ghost_prim_vars);
  *divergence_cleaning_field = get<DivergenceCleaningField>(ghost_prim_vars);

  if (need_tags_for_fluxes) {
    Variables<fluxes_tags> outermost_fluxes_vars{
        std::next(ghost_prim_vars.data(),
                  static_cast<std::ptrdiff_t>(ghost_prim_vars.size())),
        num_face_pts * buffer_size_for_fluxes};
    Variables<fluxes_tags> ghost_fluxes_vars{
        std::next(outermost_fluxes_vars.data(),
                  static_cast<std::ptrdiff_t>(outermost_fluxes_vars.size())),
        num_face_pts * buffer_size_for_fluxes * ghost_zone_size};

    for (size_t k = 0; k < ghost_zone_size; ++k) {
      const size_t interior_ix = ghost_zone_size - 1 - k;

      get<Pressure>(outermost_fluxes_vars) = data_on_slice(
          interior_pressure, subcell_extents, dim_direction, interior_ix);
      get<SpecificInternalEnergy>(outermost_fluxes_vars) =
          data_on_slice(interior_specific_internal_energy, subcell_extents,
                        dim_direction, interior_ix);
      get<SpatialMetric>(outermost_fluxes_vars) = data_on_slice(
          interior_spatial_metric, subcell_extents, dim_direction, interior_ix);
      // Components with exactly one index in dim_direction are odd under the
      // cartoon reflection and must be negated.
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j <= i; ++j) {
          if ((i == dim_direction) != (j == dim_direction)) {
            get<SpatialMetric>(outermost_fluxes_vars).get(i, j) *= -1.0;
          }
        }
      }
      get<Lapse>(outermost_fluxes_vars) = data_on_slice(
          interior_lapse, subcell_extents, dim_direction, interior_ix);
      get<Shift>(outermost_fluxes_vars) = data_on_slice(
          interior_shift, subcell_extents, dim_direction, interior_ix);
      // The x component of the shift is odd under the cartoon reflection
      get<0>(get<Shift>(outermost_fluxes_vars)) *= -1.0;

      add_slice_to_data(make_not_null(&ghost_fluxes_vars),
                        outermost_fluxes_vars, ghost_data_extents,
                        dim_direction, k);
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
