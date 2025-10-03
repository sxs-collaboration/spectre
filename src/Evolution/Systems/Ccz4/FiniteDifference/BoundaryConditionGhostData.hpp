// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

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
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/Ccz4/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace Ccz4::fd {
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
  static void apply(gsl::not_null<db::DataBox<DbTagsList>*> box,
                    const Element<3>& element,
                    const Reconstructor& reconstructor);

 private:
  template <typename FdBoundaryConditionHelper, typename DbTagsList,
            typename... FdBoundaryConditionArgsTags>
  // A helper function for calling fd_ghost() of BoundaryCondition subclasses
  static void apply_subcell_boundary_condition_impl(
      FdBoundaryConditionHelper& fd_boundary_condition_helper,
      const gsl::not_null<db::DataBox<DbTagsList>*>& box,
      tmpl::list<FdBoundaryConditionArgsTags...> /*unused*/) {
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
  using variables_for_reconstruction = typename System::variables_tag_list;

  size_t num_evolved_tensor_components = 0;
  tmpl::for_each<variables_for_reconstruction>(
        [&num_evolved_tensor_components]<typename Tag>
            (const tmpl::type_<Tag> /*meta*/) {
            num_evolved_tensor_components += Tag::type::size();
        });

  for (const auto& direction : element.external_boundaries()) {
    const auto& boundary_condition_at_direction =
        *external_boundary_condition.at(direction);

    const size_t num_face_pts{
        subcell_mesh.extents().slice_away(direction.dimension()).product()};

    // Allocate a vector to store the computed FD ghost data and assign a
    // non-owning Variables on it.
    const size_t evolved_vars_size =
        num_evolved_tensor_components * ghost_zone_size * num_face_pts;
    const size_t fluxes_size = 0;  // CCZ4 doesn't use flux variables

    auto& all_ghost_data = db::get_mutable_reference<
        evolution::dg::subcell::Tags::GhostDataForReconstruction<3>>(box);
    // Put the computed ghost data into neighbor data with {direction,
    // ElementId::external_boundary_id()} as the mortar_id key
    const DirectionalId<3> mortar_id{direction,
                                     ElementId<3>::external_boundary_id()};

    all_ghost_data[mortar_id] = evolution::dg::subcell::GhostData{1};
    DataVector& boundary_ghost_data =
        all_ghost_data.at(mortar_id).neighbor_ghost_data_for_reconstruction();
    boundary_ghost_data.destructive_resize(evolved_vars_size + fluxes_size);
    Variables<variables_for_reconstruction> ghost_data_vars{
        boundary_ghost_data.data(), evolved_vars_size};

    // We don't need to care about boundary ghost data when using the periodic
    // condition, so exclude it from the type list
    using factory_classes =
        typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            *box))>::factory_creation::factory_classes;
    using derived_boundary_conditions_for_subcell = tmpl::remove_if<
        tmpl::at<factory_classes, BoundaryConditions::BoundaryCondition>,
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
                        evolution::BoundaryConditions::Type::Ghost) {
            const auto apply_fd_ghost =
                [&boundary_condition, &direction,
                 &ghost_data_vars](const auto&... boundary_ghost_data_args) {
                  (*boundary_condition)
                      .fd_ghost(
                          make_not_null(
                              &get<
                                  ::Ccz4::Tags::ConformalMetric<DataVector, 3>>(
                                  ghost_data_vars)),
                          make_not_null(&get<gr::Tags::Lapse<DataVector>>(
                              ghost_data_vars)),
                          make_not_null(&get<gr::Tags::Shift<DataVector, 3>>(
                              ghost_data_vars)),
                          make_not_null(
                              &get<::Ccz4::Tags::ConformalFactor<DataVector>>(
                                  ghost_data_vars)),
                          make_not_null(
                              &get<::Ccz4::Tags::ATilde<DataVector, 3>>(
                                  ghost_data_vars)),
                          make_not_null(&get<gr::Tags::TraceExtrinsicCurvature<
                                            DataVector>>(ghost_data_vars)),
                          make_not_null(&get<::Ccz4::Tags::Theta<DataVector>>(
                              ghost_data_vars)),
                          make_not_null(
                              &get<::Ccz4::Tags::GammaHat<DataVector, 3>>(
                                  ghost_data_vars)),
                          make_not_null(
                              &get<
                                  ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(
                                  ghost_data_vars)),
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
  }
}
}  // namespace Ccz4::fd
