// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Utilities/ApplyAt.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {

namespace detail {
// Return the `BoundaryConditionClasses`, or get the list of derived classes
// from `Metavariables::factory_creation` if `BoundaryConditionClasses` is
// empty.
template <typename Base, typename BoundaryConditionClasses, typename DbTagsList>
struct GetBoundaryConditionClasses {
  using type = BoundaryConditionClasses;
};
template <typename Base, typename DbTagsList>
struct GetBoundaryConditionClasses<Base, tmpl::list<>, DbTagsList> {
  using type = tmpl::at<
      typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
          std::declval<db::DataBox<DbTagsList>>()))>::factory_creation::
          factory_classes,
      Base>;
};
}  // namespace detail

/*!
 * \brief Apply the `boundary_condition` to the `fields_and_fluxes` with
 * arguments from interface tags in the DataBox.
 *
 * This functions assumes the arguments for the `boundary_condition` are stored
 * in the DataBox in tags `domain::Tags::Faces<Dim, Tag>`.
 * This may turn out not to be the most efficient setup, so code that
 * uses the boundary conditions doesn't have to use this function but can
 * procure the arguments differently. For example, future optimizations may
 * involve storing a subset of arguments that don't change during an elliptic
 * solve in direction-maps in the DataBox, and slicing other arguments to the
 * interface every time the boundary conditions are applied.
 *
 * The `ArgsTransform` template parameter can be used to transform the set of
 * argument tags for the boundary conditions further. It must be compatible with
 * `tmpl::transform`. For example, it may wrap the tags in another prefix. Set
 * it to `void` (default) to apply no transformation.
 *
 * The `BoundaryConditionClasses` can be used to list a set of classes derived
 * from `elliptic::BoundaryConditions::BoundaryCondition` that are iterated to
 * determine the concrete type of `boundary_condition`. It can be `tmpl::list<>`
 * (default) to use the classes listed in `Metavariables::factory_creation`
 * instead.
 */
template <bool Linearized, typename ArgsTransform = void,
          typename BoundaryConditionClasses = tmpl::list<>, size_t Dim,
          typename DbTagsList, typename MapKeys, typename... FieldsAndFluxes>
void apply_boundary_condition(
    const elliptic::BoundaryConditions::BoundaryCondition<Dim>&
        boundary_condition,
    const db::DataBox<DbTagsList>& box, const MapKeys& map_keys_to_direction,
    FieldsAndFluxes&&... fields_and_fluxes) {
  using boundary_condition_classes =
      typename detail::GetBoundaryConditionClasses<
          elliptic::BoundaryConditions::BoundaryCondition<Dim>,
          BoundaryConditionClasses, DbTagsList>::type;
  call_with_dynamic_type<void, boundary_condition_classes>(
      &boundary_condition, [&map_keys_to_direction, &box,
                            &fields_and_fluxes...](const auto* const derived) {
        using Derived = std::decay_t<std::remove_pointer_t<decltype(derived)>>;
        using volume_tags =
            tmpl::conditional_t<Linearized,
                                typename Derived::volume_tags_linearized,
                                typename Derived::volume_tags>;
        using argument_tags = domain::make_faces_tags<
            Dim,
            tmpl::conditional_t<Linearized,
                                typename Derived::argument_tags_linearized,
                                typename Derived::argument_tags>,
            volume_tags>;
        using argument_tags_transformed =
            tmpl::conditional_t<std::is_same_v<ArgsTransform, void>,
                                argument_tags,
                                tmpl::transform<argument_tags, ArgsTransform>>;
        using volume_tags_transformed =
            tmpl::conditional_t<std::is_same_v<ArgsTransform, void>,
                                volume_tags,
                                tmpl::transform<volume_tags, ArgsTransform>>;
        elliptic::util::apply_at<argument_tags_transformed,
                                 volume_tags_transformed>(
            [&derived, &fields_and_fluxes...](const auto&... args) {
              if constexpr (Linearized) {
                derived->apply_linearized(
                    std::forward<FieldsAndFluxes>(fields_and_fluxes)...,
                    args...);
              } else {
                derived->apply(
                    std::forward<FieldsAndFluxes>(fields_and_fluxes)...,
                    args...);
              }
            },
            box, map_keys_to_direction);
      });
}
}  // namespace elliptic
