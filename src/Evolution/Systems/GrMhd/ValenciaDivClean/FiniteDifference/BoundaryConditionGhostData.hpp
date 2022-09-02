// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
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
  const ::Domain<3>& domain = db::get<domain::Tags::Domain<3>>(*box);
  const auto& external_boundary_condition =
      domain.blocks()[element.id().block_id()].external_boundary_conditions();

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
  using Pressure = hydro::Tags::Pressure<DataVector>;
  using LorentzFactorTimesSpatialVelocity =
      hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>;
  using MagneticField = hydro::Tags::MagneticField<DataVector, 3>;
  using DivergenceCleaningField =
      hydro::Tags::DivergenceCleaningField<DataVector>;

  using prims_for_reconstruction =
      tmpl::list<RestMassDensity, Pressure, LorentzFactorTimesSpatialVelocity,
                 MagneticField, DivergenceCleaningField>;

  for (const auto& direction : element.external_boundaries()) {
    ASSERT(external_boundary_condition.at(direction) != nullptr,
           "Boundary condition is not set (the pointer is null) in block "
               << element.id().block_id() << ", direction: " << direction
               << " (ValenciaDivClean::fd::BoundaryConditionGhostData)");

    const auto& boundary_condition_at_direction =
        *external_boundary_condition.at(direction);

    const size_t num_face_pts{
        subcell_mesh.extents().slice_away(direction.dimension()).product()};

    // a Variables object to store the computed FD ghost data
    Variables<prims_for_reconstruction> ghost_data_vars{ghost_zone_size *
                                                        num_face_pts};

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
        [&boundary_condition_at_direction, &box, &direction,
         &ghost_data_vars](const auto& derived_boundary_condition_v) {
          using BoundaryCondition =
              std::decay_t<decltype(*derived_boundary_condition_v)>;
          if (const auto* boundary_condition =
                  dynamic_cast<const BoundaryCondition*>(
                      &boundary_condition_at_direction);
              boundary_condition != nullptr) {
            using bcondition_interior_evolved_vars_tags =
                typename BoundaryCondition::fd_interior_evolved_variables_tags;
            using bcondition_interior_temporary_tags =
                typename BoundaryCondition::fd_interior_temporary_tags;
            using bcondition_interior_primitive_vars_tags =
                typename BoundaryCondition::
                    fd_interior_primitive_variables_tags;
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
                  [&boundary_condition, &direction,
                   &ghost_data_vars](const auto&... boundary_ghost_data_args) {
                    (*boundary_condition)
                        .fd_ghost(
                            make_not_null(
                                &get<RestMassDensity>(ghost_data_vars)),
                            make_not_null(&get<Pressure>(ghost_data_vars)),
                            make_not_null(
                                &get<LorentzFactorTimesSpatialVelocity>(
                                    ghost_data_vars)),
                            make_not_null(&get<MagneticField>(ghost_data_vars)),
                            make_not_null(
                                &get<DivergenceCleaningField>(ghost_data_vars)),
                            direction, boundary_ghost_data_args...);
                  };
              apply_subcell_boundary_condition_impl(apply_fd_ghost, box,
                                                    bcondition_interior_tags{});
            } else if constexpr (BoundaryCondition::bc_type ==
                                 evolution::BoundaryConditions::Type::Outflow) {
              // Outflow boundary condition only checks if the characteristic
              // speed is directed out of the element.
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

              const auto apply_fd_outflow =
                  [&boundary_condition, &direction, &face_mesh_velocity,
                   &ghost_data_vars, &outward_directed_normal_covector](
                      const auto&... boundary_ghost_data_args) {
                    return (*boundary_condition)
                        .fd_outflow(
                            make_not_null(
                                &get<RestMassDensity>(ghost_data_vars)),
                            make_not_null(&get<Pressure>(ghost_data_vars)),
                            make_not_null(
                                &get<LorentzFactorTimesSpatialVelocity>(
                                    ghost_data_vars)),
                            make_not_null(&get<MagneticField>(ghost_data_vars)),
                            make_not_null(
                                &get<DivergenceCleaningField>(ghost_data_vars)),
                            direction, face_mesh_velocity,
                            outward_directed_normal_covector,
                            boundary_ghost_data_args...);
                  };
              apply_subcell_boundary_condition_impl(apply_fd_outflow, box,
                                                    bcondition_interior_tags{});

              return;
            } else {
              ERROR("Unsupported boundary condition "
                    << pretty_type::short_name<BoundaryCondition>()
                    << " when using finite-difference");
            }
          }
        });

    // Put the computed ghost data into neighbor data with {direction,
    // ElementId::external_boundary_id()} as the mortar_id key
    std::vector<double> boundary_ghost_data{
        ghost_data_vars.data(),
        ghost_data_vars.data() + ghost_data_vars.size()};
    const std::pair mortar_id{direction, ElementId<3>::external_boundary_id()};

    db::mutate<evolution::dg::subcell::Tags::NeighborDataForReconstruction<3>>(
        box, [&mortar_id, &boundary_ghost_data](auto neighbor_data) {
          (*neighbor_data)[mortar_id] = boundary_ghost_data;
        });
  }
}
}  // namespace grmhd::ValenciaDivClean::fd
