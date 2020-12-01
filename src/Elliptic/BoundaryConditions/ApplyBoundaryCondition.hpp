// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Utilities/ApplyAt.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
/*!
 * \brief Apply the `boundary_condition` to the `fields_and_fluxes` with
 * arguments from interface tags in the DataBox.
 *
 * This functions assumes the arguments for the `boundary_condition` are stored
 * in the DataBox in tags
 * `domain::Tags::Interface<domain::Tags::BoundaryDirectionsInterior<Dim>,
 * Tag>`. This may turn out not to be the most efficient setup, so code that
 * uses the boundary conditions doesn't have to use this function but can
 * procure the arguments differently. For example, future optimizations may
 * involve storing a subset of arguments that don't change during an elliptic
 * solve in direction-maps in the DataBox, and slicing other arguments to the
 * interface every time the boundary conditions are applied.
 *
 * Doesn't currently support:
 * volume tag on overlaps -> either all or none of the map keys are applied
 */
template <bool Linearized, typename ArgsTransform, size_t Dim,
          typename Registrars, typename DbTagsList, typename MapKeys,
          typename... FieldsAndFluxes>
void apply_boundary_condition(
    const elliptic::BoundaryConditions::BoundaryCondition<Dim, Registrars>&
        boundary_condition,
    const db::DataBox<DbTagsList>& box, const MapKeys& map_keys_to_direction,
    const gsl::not_null<FieldsAndFluxes*>... fields_and_fluxes) noexcept {
  call_with_dynamic_type<
      void, typename elliptic::BoundaryConditions::BoundaryCondition<
                Dim, Registrars>::creatable_classes>(
      &boundary_condition,
      [&map_keys_to_direction, &box,
       &fields_and_fluxes...](const auto* const derived) noexcept {
        using Derived = std::decay_t<std::remove_pointer_t<decltype(derived)>>;
        using volume_tags =
            tmpl::conditional_t<Linearized,
                                typename Derived::volume_tags_linearized,
                                typename Derived::volume_tags>;
        using argument_tags = tmpl::transform<
            tmpl::conditional_t<Linearized,
                                typename Derived::argument_tags_linearized,
                                typename Derived::argument_tags>,
            make_interface_tag<
                tmpl::_1,
                tmpl::pin<domain::Tags::BoundaryDirectionsInterior<Dim>>,
                tmpl::pin<volume_tags>>>;
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
            [&derived, &fields_and_fluxes...](const auto&... args) noexcept {
              if constexpr (Linearized) {
                derived->apply_linearized(args..., fields_and_fluxes...);
              } else {
                derived->apply(args..., fields_and_fluxes...);
              }
            },
            box, map_keys_to_direction);
      });
}
}  // namespace elliptic
