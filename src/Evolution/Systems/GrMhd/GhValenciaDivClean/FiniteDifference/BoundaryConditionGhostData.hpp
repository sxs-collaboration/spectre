// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::fd {
/*!
 * \brief Computes finite difference ghost data for external boundary
 * conditions.
 *
 * If the element is at the external boundary, computes FD ghost data with a
 * given boundary condition and stores it into neighbor data with {direction,
 * ElementId::external_boundary_id()} as the mortar_id key.
 *
 * \note Subcell needs to be enabled for boundary elements. Otherwise this
 * function would be never called.
 *
 */
template <typename System>
struct BoundaryConditionGhostData {
  template <typename DbTagsList>
  static void apply(
      gsl::not_null<db::DataBox<DbTagsList>*> box, const Element<3>& element,
      const Reconstructor<System>& reconstructor);

 private:
  template <typename FdBoundaryConditionHelper, typename DbTagsList,
            typename... FdBoundaryConditionArgsTags>
  // A helper function for calling fd_ghost() of BoundaryCondition subclasses
  static void apply_subcell_boundary_condition_impl(
      FdBoundaryConditionHelper& fd_boundary_condition_helper,
      const gsl::not_null<db::DataBox<DbTagsList>*>& box,
      tmpl::list<FdBoundaryConditionArgsTags...>) {
    return fd_boundary_condition_helper(
        db::get<FdBoundaryConditionArgsTags>(*box)...);
  }
};

template <typename System>
template <typename DbTagsList>
void BoundaryConditionGhostData<System>::apply(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const Element<3>& element,
    const Reconstructor<System>& reconstructor) {
  const auto& external_boundary_condition =
      db::get<domain::Tags::ExternalBoundaryConditions<3>>(*box).at(
          element.id().block_id());

  // Check if the element is on the external boundary. If not, the caller is
  // doing something wrong (e.g. trying to compute FD ghost data with boundary
  // conditions at an element which is not on the external boundary).
  ASSERT(not element.external_boundaries().empty(),
         "The element (ID : " << element.id()
                              << ") is not on external boundaries");

  const Mesh<3> subcell_mesh =
      db::get<evolution::dg::subcell::Tags::Mesh<3>>(*box);

  const size_t ghost_zone_size{reconstructor.ghost_zone_size()};

  // Tags and tags list for FD reconstruction
  using RestMassDensity = hydro::Tags::RestMassDensity<DataVector>;
  using ElectronFraction = hydro::Tags::ElectronFraction<DataVector>;
  using Temperature = hydro::Tags::Temperature<DataVector>;
  using LorentzFactorTimesSpatialVelocity =
      hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>;
  using MagneticField = hydro::Tags::MagneticField<DataVector, 3>;
  using DivergenceCleaningField =
      hydro::Tags::DivergenceCleaningField<DataVector>;
  using SpacetimeMetric = gr::Tags::SpacetimeMetric<DataVector, 3>;
  using Pi = gh::Tags::Pi<DataVector, 3>;
  using Phi = gh::Tags::Phi<DataVector, 3>;

  using reconstruction_tags = GhValenciaDivClean::Tags::
      primitive_grmhd_and_spacetime_reconstruction_tags;
  using NeighborVariables = Variables<reconstruction_tags>;
  constexpr size_t number_of_tensor_components =
      NeighborVariables::number_of_independent_components;

  for (const auto& direction : element.external_boundaries()) {
    const auto& boundary_condition_at_direction =
        *external_boundary_condition.at(direction);

    const size_t num_face_pts{
        subcell_mesh.extents().slice_away(direction.dimension()).product()};

    // Allocate a vector to store the computed FD ghost data and assign a
    // non-owning Variables on it.
    auto& all_ghost_data = db::get_mutable_reference<
        evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box);
    // Put the computed ghost data into neighbor data with {direction,
    // ElementId::external_boundary_id()} as the mortar_id key
    const DirectionalId<3> mortar_id{direction,
                                     ElementId<3>::external_boundary_id()};
    all_ghost_data[mortar_id] = evolution::dg::subcell::GhostData{1};
    DataVector& boundary_ghost_data =
        all_ghost_data.at(mortar_id).neighbor_ghost_data_for_reconstruction();
    boundary_ghost_data.destructive_resize(number_of_tensor_components *
                                           ghost_zone_size * num_face_pts);
    Variables<reconstruction_tags> ghost_data_vars{boundary_ghost_data.data(),
                                                   boundary_ghost_data.size()};

    // We don't need to care about boundary ghost data when using the periodic
    // condition, so exclude it from the type list
    using factory_classes =
        typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            *box))>::factory_creation::factory_classes;
    using derived_boundary_conditions_for_subcell = tmpl::remove_if<
        tmpl::at<factory_classes, typename System::
                                      boundary_conditions_base>,
        tmpl::or_<
            std::is_base_of<domain::BoundaryConditions::MarkAsPeriodic,
                            tmpl::_1>,
            std::is_base_of<domain::BoundaryConditions::MarkAsNone, tmpl::_1>>>;

    // Now apply subcell boundary conditions
    call_with_dynamic_type<void, derived_boundary_conditions_for_subcell>(
        &boundary_condition_at_direction,
        [&box, &direction, &ghost_data_vars](const auto* boundary_condition) {
          using BoundaryCondition = std::decay_t<decltype(*boundary_condition)>;
          using bcondition_interior_evolved_vars_tags =
              typename BoundaryCondition::fd_interior_evolved_variables_tags;
          using bcondition_interior_temporary_tags =
              typename BoundaryCondition::fd_interior_temporary_tags;
          using bcondition_interior_primitive_vars_tags =
              typename BoundaryCondition::fd_interior_primitive_variables_tags;
          using bcondition_gridless_tags =
              typename BoundaryCondition::fd_gridless_tags;

          using bcondition_interior_tags =
              tmpl::append<bcondition_interior_evolved_vars_tags,
                           bcondition_interior_temporary_tags,
                           bcondition_interior_primitive_vars_tags,
                           bcondition_gridless_tags>;

          if constexpr (BoundaryCondition::bc_type ==
                            evolution::BoundaryConditions::Type::Ghost or
                        BoundaryCondition::bc_type ==
                            evolution::BoundaryConditions::Type::
                                GhostAndTimeDerivative) {
            const auto apply_fd_ghost =
                [&boundary_condition, &direction,
                 &ghost_data_vars](const auto&... boundary_ghost_data_args) {
                  (*boundary_condition)
                      .fd_ghost(
                          make_not_null(&get<SpacetimeMetric>(ghost_data_vars)),
                          make_not_null(&get<Pi>(ghost_data_vars)),
                          make_not_null(&get<Phi>(ghost_data_vars)),
                          make_not_null(&get<RestMassDensity>(ghost_data_vars)),
                          make_not_null(
                              &get<ElectronFraction>(ghost_data_vars)),
                          make_not_null(&get<Temperature>(ghost_data_vars)),
                          make_not_null(&get<LorentzFactorTimesSpatialVelocity>(
                              ghost_data_vars)),
                          make_not_null(&get<MagneticField>(ghost_data_vars)),
                          make_not_null(
                              &get<DivergenceCleaningField>(ghost_data_vars)),
                          direction, boundary_ghost_data_args...);
                };
            apply_subcell_boundary_condition_impl(apply_fd_ghost, box,
                                                  bcondition_interior_tags{});
          } else {
            ERROR("Unsupported boundary condition "
                  << pretty_type::short_name<BoundaryCondition>()
                  << " when using finite-difference");
          }
        });
    if (dynamic_cast<const BoundaryConditions::DirichletAnalytic<System>*>(
            &boundary_condition_at_direction) != nullptr and
        reconstructor.reconstruct_rho_times_temperature()) {
      // If we reconstruct rho*T we end up having to divide by rho to compute
      // T. In some cases, like when evolving a TOV star with an analytic
      // boundary condition, the boundary condition sets rho=0. While
      // unphysical in general, this is how we have implemented the
      // solutions. We deal with this by applying our atmosphere treatment to
      // the reconstructed stated.

      const auto& atmosphere_fixer =
          db::get<::Tags::VariableFixer<VariableFixing::FixToAtmosphere<3>>>(
              *box);
      const auto& equation_of_state =
          db::get<hydro::Tags::GrmhdEquationOfState>(*box);

      tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{};
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j) {
          spatial_metric.get(i, j).set_data_ref(
              &get<SpacetimeMetric>(ghost_data_vars).get(i + 1, j + 1));
        }
      }

      Variables<tmpl::list<hydro::Tags::SpatialVelocity<DataVector, 3>,
                           hydro::Tags::LorentzFactor<DataVector>,
                           hydro::Tags::SpecificInternalEnergy<DataVector>,
                           hydro::Tags::Pressure<DataVector>>>
          temp_hydro_vars{ghost_data_vars.number_of_grid_points()};

      const auto& lorentz_factor_times_spatial_velocity =
          get<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>(
              ghost_data_vars);
      auto& lorentz_factor =
          get<hydro::Tags::LorentzFactor<DataVector>>(temp_hydro_vars);
      get(lorentz_factor) = 0.0;
      for (size_t i = 0; i < 3; ++i) {
        get(lorentz_factor) +=
            spatial_metric.get(i, i) *
            square(lorentz_factor_times_spatial_velocity.get(i));
        for (size_t j = i + 1; j < 3; ++j) {
          get(lorentz_factor) += 2.0 * spatial_metric.get(i, j) *
                                 lorentz_factor_times_spatial_velocity.get(i) *
                                 lorentz_factor_times_spatial_velocity.get(j);
        }
      }
      get(lorentz_factor) = sqrt(1.0 + get(lorentz_factor));
      auto& spatial_velocity =
          get<hydro::Tags::SpatialVelocity<DataVector, 3>>(temp_hydro_vars) =
              lorentz_factor_times_spatial_velocity;
      for (size_t i = 0; i < 3; ++i) {
        spatial_velocity.get(i) /= get(lorentz_factor);
      }

      get<hydro::Tags::SpecificInternalEnergy<DataVector>>(temp_hydro_vars) =
          equation_of_state
              .specific_internal_energy_from_density_and_temperature(
                  get<RestMassDensity>(ghost_data_vars),
                  get<Temperature>(ghost_data_vars),
                  get<ElectronFraction>(ghost_data_vars));
      get<hydro::Tags::Pressure<DataVector>>(temp_hydro_vars) =
          equation_of_state.pressure_from_density_and_temperature(
              get<RestMassDensity>(ghost_data_vars),
              get<Temperature>(ghost_data_vars),
              get<ElectronFraction>(ghost_data_vars));

      atmosphere_fixer(
          make_not_null(&get<RestMassDensity>(ghost_data_vars)),
          make_not_null(&get<hydro::Tags::SpecificInternalEnergy<DataVector>>(
              temp_hydro_vars)),
          make_not_null(&get<hydro::Tags::SpatialVelocity<DataVector, 3>>(
              temp_hydro_vars)),
          make_not_null(
              &get<hydro::Tags::LorentzFactor<DataVector>>(temp_hydro_vars)),
          make_not_null(
              &get<hydro::Tags::Pressure<DataVector>>(temp_hydro_vars)),
          make_not_null(&get<Temperature>(ghost_data_vars)),
          get<ElectronFraction>(ghost_data_vars), spatial_metric,
          equation_of_state);
    }
  }  // for (direction : external boundaries)
}
}  // namespace grmhd::GhValenciaDivClean::fd
