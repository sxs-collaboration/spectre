// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/NeighborData.hpp"
#include "Evolution/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::fd {
/*!
 * \brief Computes finite difference ghost data for external boundary
 * conditions.
 *
 */
struct BoundaryGhostData {
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

  template <typename DbTagsList, size_t Dim>
  static void apply(const gsl::not_null<db::DataBox<DbTagsList>*> box,
                    const Element<Dim>& element,
                    const Reconstructor<Dim>& reconstructor);
};

template <typename DbTagsList, size_t Dim>
void BoundaryGhostData::apply(const gsl::not_null<db::DataBox<DbTagsList>*> box,
                              const Element<Dim>& element,
                              const Reconstructor<Dim>& reconstructor) {
  const ::Domain<Dim>& domain = db::get<domain::Tags::Domain<Dim>>(*box);
  const auto& external_boundary_condition =
      domain.blocks()[element.id().block_id()].external_boundary_conditions();

  const Mesh<Dim> subcell_mesh =
      db::get<evolution::dg::subcell::Tags::Mesh<Dim>>(*box);

  const size_t ghost_zone_size{reconstructor.ghost_zone_size()};

  // Tags for reconstruction
  using prims_to_reconstruct =
      tmpl::list<NewtonianEuler::Tags::MassDensity<DataVector>,
                 NewtonianEuler::Tags::Velocity<DataVector, Dim>,
                 NewtonianEuler::Tags::Pressure<DataVector>>;

  for (const auto& direction : Direction<Dim>::all_directions()) {
    if (not(element.neighbors().contains(direction))) {
      // If a direction is not in elements.neighbors(), then the element must
      // be at the external boundary
      ASSERT(element.external_boundaries().count(direction),
             "Element contains a direction which is neither neighbor nor "
             "external boundary");
      // Also check if the pointer is not null. This will catch an error that
      // the pointer has a correct type but not pointing anything
      ASSERT(external_boundary_condition.at(direction) != nullptr, "");
      const auto& boundary_condition =
          *external_boundary_condition.at(direction);

      const size_t num_face_pts{
          subcell_mesh.extents().slice_away(direction.dimension()).product()};

      // a Variables object to store the computed FD ghost data
      Variables<prims_to_reconstruct> ghost_data_vars{ghost_zone_size *
                                                      num_face_pts};

      // We don't need to care about boundary ghost data when using the periodic
      // condition, so exclude it from the type list
      using factory_classes =
          typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
              *box))>::factory_creation::factory_classes;
      using derived_boundary_conditions_for_subcell = tmpl::remove_if<
          tmpl::at<factory_classes,
                   typename NewtonianEuler::BoundaryConditions::
                       BoundaryCondition<Dim>>,
          tmpl::or_<std::is_base_of<domain::BoundaryConditions::MarkAsPeriodic,
                                    tmpl::_1>,
                    std::is_base_of<domain::BoundaryConditions::MarkAsNone,
                                    tmpl::_1>>>;

      // Now apply subcell boundary conditions
      call_with_dynamic_type<void, derived_boundary_conditions_for_subcell>(
          &boundary_condition, [&boundary_condition, &box, &direction,
                                &ghost_data_vars, &num_face_pts, &subcell_mesh](
                                   const auto& derived_boundary_condition_v) {
            using BoundaryCondition =
                std::decay_t<decltype(*derived_boundary_condition_v)>;
            if (dynamic_cast<const BoundaryCondition*>(&boundary_condition) !=
                nullptr) {
              using bcondition_interior_evolved_vars_tags =
                  typename BoundaryCondition::
                      fd_interior_evolved_variables_tags;
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
                auto apply_fd_ghost =
                    [&boundary_condition, &direction, &ghost_data_vars](
                        const auto&... boundary_ghost_data_args) {
                      dynamic_cast<const BoundaryCondition&>(boundary_condition)
                          .fd_ghost(
                              make_not_null(
                                  &get<NewtonianEuler::Tags::MassDensity<
                                      DataVector>>(ghost_data_vars)),
                              make_not_null(
                                  &get<NewtonianEuler::Tags::Velocity<
                                      DataVector, Dim>>(ghost_data_vars)),
                              make_not_null(&get<NewtonianEuler::Tags::Pressure<
                                                DataVector>>(ghost_data_vars)),
                              direction, boundary_ghost_data_args...);
                    };
                apply_subcell_boundary_condition_impl(
                    apply_fd_ghost, box, bcondition_interior_tags{});
              } else if constexpr (BoundaryCondition::bc_type ==
                                   evolution::BoundaryConditions::Type::
                                       Outflow) {
                // Outflow boundary condition checks if the characteristic speed
                // is directed out of the element.
                const auto& volume_mesh_velocity =
                    db::get<domain::Tags::MeshVelocity<Dim, Frame::Inertial>>(
                        *box);
                std::optional<tnsr::I<DataVector, Dim>> face_mesh_velocity{};

                if (volume_mesh_velocity.has_value()) {
                  face_mesh_velocity = tnsr::I<DataVector, Dim>{num_face_pts};
                  evolution::dg::project_tensor_to_boundary(
                      make_not_null(&*face_mesh_velocity),
                      *volume_mesh_velocity, subcell_mesh, direction);
                }

                const auto& normal_covector_and_magnitude = db::get<
                    evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>>(*box);
                auto outward_directed_normal_covector =
                    get<evolution::dg::Tags::NormalCovector<Dim>>(
                        normal_covector_and_magnitude.at(direction).value());

                const auto apply_fd_outflow =
                    [&boundary_condition, &direction, &face_mesh_velocity,
                     &ghost_data_vars, &outward_directed_normal_covector](
                        const auto&... boundary_ghost_data_args) {
                      return dynamic_cast<const BoundaryCondition&>(
                                 boundary_condition)
                          .fd_outflow(
                              make_not_null(
                                  &get<NewtonianEuler::Tags::MassDensity<
                                      DataVector>>(ghost_data_vars)),
                              make_not_null(
                                  &get<NewtonianEuler::Tags::Velocity<
                                      DataVector, Dim>>(ghost_data_vars)),
                              make_not_null(&get<NewtonianEuler::Tags::Pressure<
                                                DataVector>>(ghost_data_vars)),
                              direction, face_mesh_velocity,
                              outward_directed_normal_covector,
                              boundary_ghost_data_args...);
                    };
                apply_subcell_boundary_condition_impl(
                    apply_fd_outflow, box, bcondition_interior_tags{});
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
      const std::pair mortar_id{direction,
                                ElementId<Dim>::external_boundary_id()};

      db::mutate<
          evolution::dg::subcell::Tags::NeighborDataForReconstruction<Dim>>(
          box, [&mortar_id, &boundary_ghost_data](auto neighbor_data) {
            (*neighbor_data)[mortar_id] = boundary_ghost_data;
          });
    }
  }
}

}  // namespace NewtonianEuler::fd
