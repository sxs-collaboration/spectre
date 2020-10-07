// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/DataOnSlice.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BjorhusImpl.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/BoundaryConditions.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

namespace GeneralizedHarmonic {
namespace Actions {
namespace BoundaryConditions_detail {}  // namespace BoundaryConditions_detail

/// \ingroup ActionsGroup
/// \brief Computes contributions on the interior side from the volume, and
/// imposes constraint preserving boundary conditions on the exterior side.
///
/// With:
/// - Boundary<Tag> =
///   Tags::Interface<Tags::BoundaryDirections<volume_dim>, Tag>
/// - External<Tag> =
///   Tags::Interface<Tags::ExternalBoundaryDirections<volume_dim>, Tag>
///
/// Uses:
/// - GlobalCache:
///   - Metavariables::normal_dot_numerical_flux
///   - Metavariables::boundary_condition
/// - DataBox:
///   - Tags::Element<volume_dim>
///   - Boundary<Metavariables::normal_dot_numerical_flux::type::argument_tags>
///   - External<Metavariables::normal_dot_numerical_flux::type::argument_tags>
///   - Boundary<Tags::Mesh<volume_dim - 1>>
///   - External<Tags::Mesh<volume_dim - 1>>
///   - Boundary<Tags::Magnitude<Tags::UnnormalizedFaceNormal<volume_dim>>>,
///   - External<Tags::Magnitude<Tags::UnnormalizedFaceNormal<volume_dim>>>,
///   - Boundary<Tags::BoundaryCoordinates<volume_dim>>,
///   - Metavariables::temporal_id
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///      - External<Tags::dt<typename system::variables_tag>>
///
template <typename Metavariables>
struct ImposeBjorhusBoundaryConditions {
 public:
  using const_global_cache_tags =
      tmpl::list<typename Metavariables::normal_dot_numerical_flux,
                 typename Metavariables::boundary_condition_tag>;

 private:
  // {VSpacetimeMetric,VZero,VPlus,VMinus}BcMethod enumerate available
  // choices for boundary condition. The choice is made in the `apply`
  // method below.
  using VSpacetimeMetricBcMethod =
      BoundaryConditions_detail::VSpacetimeMetricBcMethod;
  using VZeroBcMethod = BoundaryConditions_detail::VZeroBcMethod;
  using VPlusBcMethod = BoundaryConditions_detail::VPlusBcMethod;
  using VMinusBcMethod = BoundaryConditions_detail::VMinusBcMethod;

  template <size_t VolumeDim, VSpacetimeMetricBcMethod VSpacetimeMetricMethod,
            VZeroBcMethod VZeroMethod, VPlusBcMethod VPlusMethod,
            VMinusBcMethod VMinusMethod, typename DbTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename... InboxTags>
  struct apply_impl {
    static std::tuple<db::DataBox<DbTags>&&> function_impl(
        db::DataBox<DbTags>& box,
        tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
        Parallel::GlobalCache<Metavariables>& cache,
        const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
      // Get information about system:
      // tags for evolved variables and their time derivatives
      using system = typename Metavariables::system;
      using variables_tag = typename system::variables_tag;
      using dt_variables_tag =
          db::add_tag_prefix<Metavariables::temporal_id::template step_prefix,
                             variables_tag>;
      constexpr const size_t number_of_independent_components =
          dt_variables_tag::type::number_of_independent_components;

      const db::item_type<domain::Tags::Mesh<VolumeDim>>& mesh =
          db::get<domain::Tags::Mesh<VolumeDim>>(box);
      const size_t volume_grid_points = mesh.extents().product();
      const auto& unit_normal_one_forms = db::get<domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsInterior<VolumeDim>,
          ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<
              VolumeDim, Frame::Inertial>>>>(box);
      const auto& volume_all_vars = db::get<variables_tag>(box);
      const auto& volume_all_dt_vars = db::get<dt_variables_tag>(box);
      const auto& external_bdry_char_speeds = db::get<domain::Tags::Interface<
          domain::Tags::BoundaryDirectionsInterior<VolumeDim>,
          Tags::CharacteristicSpeeds<VolumeDim, Frame::Inertial>>>(box);
      const auto& external_bdry_inertial_coords =
          db::get<domain::Tags::Interface<
              domain::Tags::BoundaryDirectionsInterior<VolumeDim>,
              domain::Tags::Coordinates<VolumeDim, Frame::Inertial>>>(box);

      // Apply boundary condition:
      // Loop over external boundaries and set dt_volume_vars on them
      for (auto& external_direction_and_normals : unit_normal_one_forms) {
        const auto& direction = external_direction_and_normals.first;
        const size_t dimension = direction.dimension();
        const auto& unit_normal_one_form =
            external_direction_and_normals.second;
        const size_t slice_grid_points =
            mesh.extents().slice_away(dimension).product();
        // Get U on this slice
        const auto vars =
            data_on_slice(volume_all_vars, mesh.extents(), dimension,
                          index_to_slice_at(mesh.extents(), direction));
        ASSERT(vars.number_of_grid_points() == slice_grid_points,
               "vars_on_slice has wrong number of grid points.  "
               "Expected "
                   << slice_grid_points << ", got "
                   << vars.number_of_grid_points());
        // Get dt<U> on this slice
        const auto dt_vars =
            data_on_slice(volume_all_dt_vars, mesh.extents(), dimension,
                          index_to_slice_at(mesh.extents(), direction));
        // Get characteristic speeds
        const auto& char_speeds = external_bdry_char_speeds.at(direction);
        // For external boundaries that are within a horizon,
        // all characteristic fields are outgoing (toward the singularity)
        if (BoundaryConditions_detail::min_characteristic_speed<VolumeDim>(
                char_speeds) >= 0.) {
          continue;
        }
        // Get boundary coordinates
        const auto& inertial_coords =
            external_bdry_inertial_coords.at(direction);

        // Create and store all temporaries needed here and elsewhere
        BoundaryConditions_detail::BjorhusIntermediatesComputer intermediates(
            slice_grid_points);
        intermediates.compute_vars(box, direction, dimension, mesh, vars,
                                   dt_vars, unit_normal_one_form, char_speeds);

        db::mutate<dt_variables_tag>(
            make_not_null(&box),
            // Function that applies bdry conditions to dt<variables>
            [
              &volume_grid_points, &slice_grid_points, &mesh, &dimension,
              &direction, &intermediates, &vars, &dt_vars,
              &unit_normal_one_form, &inertial_coords, &char_speeds
            ](const gsl::not_null<db::item_type<dt_variables_tag>*>
                  volume_dt_vars,
              const double /* time */, const auto& /* boundary_condition */
              ) noexcept {
              // Preliminaries
              ASSERT(
                  volume_dt_vars->number_of_grid_points() == volume_grid_points,
                  "volume_dt_vars has wrong number of grid points.  Expected "
                      << volume_grid_points << ", got "
                      << volume_dt_vars->number_of_grid_points());
              // Compute desired values of dt_volume_vars and set it.
              // At all points on the interface where the char speed of
              // given characteristic field is positive, we "do nothing", and
              // when its negative, we apply Bjorhus BCs. This is achieved
              // through `set_bc_when_char_speed_is_negative`.
              const auto bc_dt_u_psi =
                  BoundaryConditions_detail::set_bc_when_char_speed_is_negative(
                      intermediates.get_var(
                          Tags::VSpacetimeMetric<VolumeDim, Frame::Inertial>{}),
                      BoundaryConditions_detail::set_dt_v_psi<
                          typename Tags::VSpacetimeMetric<
                              VolumeDim, Frame::Inertial>::type,
                          VolumeDim>::apply(VSpacetimeMetricMethod,
                                            make_not_null(&intermediates), vars,
                                            dt_vars, unit_normal_one_form),
                      char_speeds.at(0));
              const auto bc_dt_u_zero =
                  BoundaryConditions_detail::set_bc_when_char_speed_is_negative(
                      intermediates.get_var(
                          Tags::VZero<VolumeDim, Frame::Inertial>{}),
                      BoundaryConditions_detail::set_dt_v_zero<
                          typename Tags::VZero<VolumeDim,
                                               Frame::Inertial>::type,
                          VolumeDim>::apply(VZeroMethod,
                                            make_not_null(&intermediates), vars,
                                            dt_vars, unit_normal_one_form),
                      char_speeds.at(1));
              const auto bc_dt_u_plus =
                  BoundaryConditions_detail::set_bc_when_char_speed_is_negative(
                      intermediates.get_var(
                          Tags::VPlus<VolumeDim, Frame::Inertial>{}),
                      BoundaryConditions_detail::set_dt_v_plus<
                          typename Tags::VPlus<VolumeDim,
                                               Frame::Inertial>::type,
                          VolumeDim>::apply(VPlusMethod,
                                            make_not_null(&intermediates), vars,
                                            dt_vars, unit_normal_one_form),
                      char_speeds.at(2));
              const auto bc_dt_u_minus =
                  BoundaryConditions_detail::set_bc_when_char_speed_is_negative(
                      intermediates.get_var(
                          Tags::VMinus<VolumeDim, Frame::Inertial>{}),
                      BoundaryConditions_detail::set_dt_v_minus<
                          typename Tags::VMinus<VolumeDim,
                                                Frame::Inertial>::type,
                          VolumeDim>::apply(VMinusMethod,
                                            make_not_null(&intermediates), vars,
                                            dt_vars, inertial_coords,
                                            unit_normal_one_form),
                      char_speeds.at(3));
              // Convert them to desired values on dt<U>
              const auto bc_dt_all_u =
                  evolved_fields_from_characteristic_fields(
                      intermediates.get_var(Tags::ConstraintGamma2{}),
                      bc_dt_u_psi, bc_dt_u_zero, bc_dt_u_plus, bc_dt_u_minus,
                      unit_normal_one_form);
              // Now store final values of dt<U> in suitable data structure
              // FIXME: Can we extract this list of dt<U> tags directly from
              // `dt_variables_tag`?
              const tuples::TaggedTuple<
                  db::add_tag_prefix<
                      Metavariables::temporal_id::template step_prefix,
                      gr::Tags::SpacetimeMetric<VolumeDim, Frame::Inertial,
                                                DataVector>>,
                  db::add_tag_prefix<
                      Metavariables::temporal_id::template step_prefix,
                      Tags::Pi<VolumeDim, Frame::Inertial>>,
                  db::add_tag_prefix<
                      Metavariables::temporal_id::template step_prefix,
                      Tags::Phi<VolumeDim, Frame::Inertial>>>
                  bc_dt_tuple(
                      std::move(get<gr::Tags::SpacetimeMetric<
                                    VolumeDim, Frame::Inertial, DataVector>>(
                          bc_dt_all_u)),
                      std::move(get<Tags::Pi<VolumeDim, Frame::Inertial>>(
                          bc_dt_all_u)),
                      std::move(get<Tags::Phi<VolumeDim, Frame::Inertial>>(
                          bc_dt_all_u)));
              const auto slice_data_ = variables_from_tagged_tuple(bc_dt_tuple);
              const auto* slice_data = slice_data_.data();

              // Assign BC values of dt_volume_vars on external boundary
              // slices of volume variables
              auto* const volume_dt_data = volume_dt_vars->data();
              for (SliceIterator si(
                       mesh.extents(), dimension,
                       index_to_slice_at(mesh.extents(), direction));
                   si; ++si) {
                for (size_t i = 0; i < number_of_independent_components; ++i) {
                  // clang-tidy: do not use pointer arithmetic
                  volume_dt_data[si.volume_offset() +       // NOLINT
                                 i * volume_grid_points] =  // NOLINT
                      slice_data[si.slice_offset() +        // NOLINT
                                 i * slice_grid_points];    // NOLINT
                }
              }
            },
            db::get<::Tags::Time>(box),
            get<typename Metavariables::boundary_condition_tag>(cache));
      }

      return std::forward_as_tuple(std::move(box));
    }
  };

 public:
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList action_list,
      const ParallelComponent* const parallel_component) noexcept {
    return apply_impl<Metavariables::system::volume_dim,
                      // BC choice for U_\Psi
                      Metavariables::boundary_conditions::v_spacetime_bc_method,
                      // BC choice for U_0
                      Metavariables::boundary_conditions::v_zero_bc_method,
                      // BC choice for U_+
                      Metavariables::boundary_conditions::v_plus_bc_method,
                      // BC choice for U_-
                      Metavariables::boundary_conditions::v_minus_bc_method,
                      DbTags, ArrayIndex, ActionList, ParallelComponent,
                      InboxTags...>::function_impl(box, inboxes, cache,
                                                   array_index, action_list,
                                                   parallel_component);
  }
};

}  // namespace Actions
}  // namespace GeneralizedHarmonic
