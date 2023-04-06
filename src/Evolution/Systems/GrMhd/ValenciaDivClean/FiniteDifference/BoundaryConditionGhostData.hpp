// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
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
#include "Evolution/DgSubcell/Tags/CellCenteredFlux.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::fd {
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
struct BoundaryConditionGhostData {
  template <typename DbTagsList>
  static void apply(const gsl::not_null<db::DataBox<DbTagsList>*> box,
                    const Element<3>& element,
                    const Reconstructor& reconstructor);

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

template <typename DbTagsList>
void BoundaryConditionGhostData::apply(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const Element<3>& element, const Reconstructor& reconstructor) {
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
  using Pressure = hydro::Tags::Pressure<DataVector>;
  using LorentzFactorTimesSpatialVelocity =
      hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>;
  using MagneticField = hydro::Tags::MagneticField<DataVector, 3>;
  using DivergenceCleaningField =
      hydro::Tags::DivergenceCleaningField<DataVector>;

  using prims_for_reconstruction =
      tmpl::list<RestMassDensity, ElectronFraction, Pressure,
                 LorentzFactorTimesSpatialVelocity, MagneticField,
                 DivergenceCleaningField>;

  size_t num_prims_tensor_components = 0;
  tmpl::for_each<prims_for_reconstruction>([&num_prims_tensor_components](
                                               auto tag) {
    num_prims_tensor_components += tmpl::type_from<decltype(tag)>::type::size();
  });

  using flux_variables =
      typename grmhd::ValenciaDivClean::System::flux_variables;
  const bool compute_cell_centered_flux =
      db::get<
          evolution::dg::subcell::Tags::CellCenteredFlux<flux_variables, 3>>(
          *box)
          .has_value();

  for (const auto& direction : element.external_boundaries()) {
    const auto& boundary_condition_at_direction =
        *external_boundary_condition.at(direction);

    const size_t num_face_pts{
        subcell_mesh.extents().slice_away(direction.dimension()).product()};

    // Allocate a vector to store the computed FD ghost data and assign a
    // non-owning Variables on it.
    using FluxVars =
        Variables<db::wrap_tags_in<::Tags::Flux, flux_variables,
                                    tmpl::size_t<3>, Frame::Inertial>>;
    const size_t prims_size =
        num_prims_tensor_components * ghost_zone_size * num_face_pts;
    const size_t fluxes_size =
        (compute_cell_centered_flux ? FluxVars::number_of_independent_components
         : 0) *
        ghost_zone_size * num_face_pts;

    auto& all_ghost_data = db::get_mutable_reference<
        evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box);
    // Put the computed ghost data into neighbor data with {direction,
    // ElementId::external_boundary_id()} as the mortar_id key
    const std::pair mortar_id{direction, ElementId<3>::external_boundary_id()};

    all_ghost_data[mortar_id] = evolution::dg::subcell::GhostData{1};
    DataVector& boundary_ghost_data =
        all_ghost_data.at(mortar_id).neighbor_ghost_data_for_reconstruction();
    boundary_ghost_data.destructive_resize(prims_size + fluxes_size);
    Variables<prims_for_reconstruction> ghost_data_vars{
        boundary_ghost_data.data(), prims_size};
    std::optional<FluxVars> cell_centered_ghost_fluxes{};
    if (compute_cell_centered_flux) {
      cell_centered_ghost_fluxes = FluxVars{};
      cell_centered_ghost_fluxes.value().set_data_ref(
          std::next(boundary_ghost_data.data(),
                    static_cast<std::ptrdiff_t>(prims_size)),
          fluxes_size);
    }

    // We don't need to care about boundary ghost data when using the periodic
    // condition, so exclude it from the type list
    using factory_classes =
        typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            *box))>::factory_creation::factory_classes;
    using derived_boundary_conditions_for_subcell = tmpl::remove_if<
        tmpl::at<factory_classes, typename System::boundary_conditions_base>,
        tmpl::or_<
            std::is_base_of<domain::BoundaryConditions::MarkAsPeriodic,
                            tmpl::_1>,
            std::is_base_of<domain::BoundaryConditions::MarkAsNone, tmpl::_1>>>;

    // Now apply subcell boundary conditions
    call_with_dynamic_type<void, derived_boundary_conditions_for_subcell>(
        &boundary_condition_at_direction,
        [&box, &cell_centered_ghost_fluxes, &direction,
         &ghost_data_vars](const auto* boundary_condition) {
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
                        evolution::BoundaryConditions::Type::Ghost) {
            const auto apply_fd_ghost =
                [&boundary_condition, &cell_centered_ghost_fluxes, &direction,
                 &ghost_data_vars](const auto&... boundary_ghost_data_args) {
                  (*boundary_condition)
                      .fd_ghost(
                          make_not_null(&get<RestMassDensity>(ghost_data_vars)),
                          make_not_null(
                              &get<ElectronFraction>(ghost_data_vars)),
                          make_not_null(&get<Pressure>(ghost_data_vars)),
                          make_not_null(&get<LorentzFactorTimesSpatialVelocity>(
                              ghost_data_vars)),
                          make_not_null(&get<MagneticField>(ghost_data_vars)),
                          make_not_null(
                              &get<DivergenceCleaningField>(ghost_data_vars)),
                          make_not_null(&cell_centered_ghost_fluxes), direction,
                          boundary_ghost_data_args...);
                };
            apply_subcell_boundary_condition_impl(apply_fd_ghost, box,
                                                  bcondition_interior_tags{});
          } else if constexpr (BoundaryCondition::bc_type ==
                               evolution::BoundaryConditions::Type::
                                   DemandOutgoingCharSpeeds) {
            // This boundary condition only checks if all the characteristic
            // speeds are directed outward.
            const auto& volume_mesh_velocity =
                db::get<domain::Tags::MeshVelocity<3, Frame::Inertial>>(*box);
            if (volume_mesh_velocity.has_value()) {
              ERROR("Subcell currently does not support moving mesh");
            }

            std::optional<tnsr::I<DataVector, 3>> face_mesh_velocity{};

            const auto& normal_covector_and_magnitude =
                db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<3>>(
                    *box);
            const auto outward_directed_normal_covector =
                get<evolution::dg::Tags::NormalCovector<3>>(
                    normal_covector_and_magnitude.at(direction).value());

            const auto apply_fd_demand_outgoing_char_speeds =
                [&boundary_condition, &cell_centered_ghost_fluxes, &direction,
                 &face_mesh_velocity, &ghost_data_vars,
                 &outward_directed_normal_covector](
                    const auto&... boundary_ghost_data_args) {
                  return (*boundary_condition)
                      .fd_demand_outgoing_char_speeds(
                          make_not_null(&get<RestMassDensity>(ghost_data_vars)),
                          make_not_null(
                              &get<ElectronFraction>(ghost_data_vars)),
                          make_not_null(&get<Pressure>(ghost_data_vars)),
                          make_not_null(&get<LorentzFactorTimesSpatialVelocity>(
                              ghost_data_vars)),
                          make_not_null(&get<MagneticField>(ghost_data_vars)),
                          make_not_null(
                              &get<DivergenceCleaningField>(ghost_data_vars)),
                          make_not_null(&cell_centered_ghost_fluxes), direction,
                          face_mesh_velocity, outward_directed_normal_covector,
                          boundary_ghost_data_args...);
                };
            apply_subcell_boundary_condition_impl(
                apply_fd_demand_outgoing_char_speeds, box,
                bcondition_interior_tags{});

            return;
          } else {
            ERROR("Unsupported boundary condition "
                  << pretty_type::short_name<BoundaryCondition>()
                  << " when using finite-difference");
          }
        });
  }
}
}  // namespace grmhd::ValenciaDivClean::fd
