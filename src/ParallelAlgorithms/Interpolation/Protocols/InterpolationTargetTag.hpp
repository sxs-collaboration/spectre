// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <type_traits>

#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeTargetPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/ComputeVarsToInterpolate.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/PostInterpolationCallback.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace intrp::protocols {
namespace detail {
// if_alias_exists_assert_conforms_to::value will always be `true` because we
// want the static assert in the protocol to pass regardless of if the
// `compute_vars_to_interpolate` alias exists in the conforming type. It's just
// that if the alias does exist, there's an additional static assert to ensure
// that the alias conforms to the ComputeVarsToInterpolate protocol
template <typename ConformingType, bool HasAlias>
struct if_alias_exists_assert_conforms_to;

template <typename ConformingType>
struct if_alias_exists_assert_conforms_to<ConformingType, false> {
  static constexpr bool value = true;
};

template <typename ConformingType>
struct if_alias_exists_assert_conforms_to<ConformingType, true> {
  static constexpr bool value = true;
  static_assert(tt::assert_conforms_to_v<
                typename ConformingType::compute_vars_to_interpolate,
                ComputeVarsToInterpolate>);
};

template <typename ConformingType, bool HasAlias>
constexpr bool if_alias_exists_assert_conforms_to_v =
    if_alias_exists_assert_conforms_to<ConformingType, HasAlias>::value;
}  // namespace detail

/*!
 * \brief A protocol for `InterpolationTargetTag`s that are used in the
 * intrp::InterpolationTarget parallel component.
 *
 * \details A struct conforming to the `InterpolationTargetTag` protocol must
 * have
 *
 * - a type alias `temporal_id` to a tag that tells the interpolation target
 *   what values of time to use (for example, `::Tags::Time`).
 *
 * - a type alias `vars_to_interpolate_to_target` which is a `tmpl::list` of
 *   tags describing variables to interpolate. Will be used to construct a
 *   `Variables`.
 *
 * - a type alias `compute_items_on_target` which is a `tmpl::list` of compute
 *   items that uses `vars_to_interpolate_to_target` as input.
 *
 * - a type alias `compute_target_points` that conforms to the
 *   intrp::protocols::ComputeTargetPoints protocol. This will compute the
 *   points on the surface that we are interpolating onto.
 *
 * - a type alias `post_interpolation_callback` that conforms to the
 *   intrp::protocols::PostInterpolationCallback protocol. After the
 *   interpolation is complete, call this struct's `apply` function.
 *
 * A struct conforming to this protocol can also optionally have
 *
 * - a type alias `compute_vars_to_interpolate` that conforms to the
 *   intrp::protocols::ComputeVarsToInterpolate protocol. This is a struct that
 *   computes quantities in the volume that are required to compute different
 *   quantities on the surface we are interpolating to.
 *
 * - a type alias `interpolating_component` to the parallel component that will
 *   be interpolating to the interpolation target. Only needed when *not* using
 *   the Interpolator ParallelComponent.
 *
 * An example of a struct that conforms to this protocol is
 *
 * \snippet Helpers/ParallelAlgorithms/Interpolation/Examples.hpp InterpolationTargetTag
 */
struct InterpolationTargetTag {
  template <typename ConformingType>
  struct test {
    using temporal_id = typename ConformingType::temporal_id;

    using vars_to_interpolate_to_target =
        typename ConformingType::vars_to_interpolate_to_target;

    // If the conforming type has a `compute_vars_to_interpolate` alias, make
    // sure it conforms to the ComputeVarsToInterpolate protocol, otherwise just
    // return true
    static_assert(detail::if_alias_exists_assert_conforms_to_v<
                  ConformingType,
                  InterpolationTarget_detail::has_compute_vars_to_interpolate_v<
                      ConformingType>>);

    using compute_items_on_target =
        typename ConformingType::compute_items_on_target;

    using compute_target_points =
        typename ConformingType::compute_target_points;
    static_assert(
        tt::assert_conforms_to_v<compute_target_points, ComputeTargetPoints>);

    using post_interpolation_callback =
        typename ConformingType::post_interpolation_callback;
    static_assert(tt::assert_conforms_to_v<post_interpolation_callback,
                                           PostInterpolationCallback>);
  };
};
}  // namespace intrp::protocols
