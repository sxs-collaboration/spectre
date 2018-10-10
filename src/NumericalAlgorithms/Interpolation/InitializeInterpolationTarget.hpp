// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp {

/// Holds Actions for Interpolator and InterpolationTarget.
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Initializes an InterpolationTarget
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds:
///   - `Tags::IndicesOfFilledInterpPoints`
///   - `Tags::TemporalIds<Metavariables>`
///   - `::Tags::Domain<VolumeDim, Frame>`
///   - `::Tags::Variables<typename
///                   InterpolationTargetTag::vars_to_interpolate_to_target>`
/// - Removes: nothing
/// - Modifies: nothing
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename InterpolationTargetTag>
struct InitializeInterpolationTarget {
  /// For requirements on Metavariables, see InterpolationTarget
  template <typename Metavariables, size_t VolumeDim, typename Frame>
  using return_tag_list = tmpl::list<
      Tags::IndicesOfFilledInterpPoints, Tags::TemporalIds<Metavariables>,
      ::Tags::Domain<VolumeDim, Frame>,
      ::Tags::Variables<
          typename InterpolationTargetTag::vars_to_interpolate_to_target>>;
  template <typename... InboxTags, typename Metavariables, typename ArrayIndex,
            typename ActionList, typename ParallelComponent, size_t VolumeDim,
            typename Frame>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    Domain<VolumeDim, Frame>&& domain) noexcept {
    return std::make_tuple(db::create<db::get_items<return_tag_list<
                               Metavariables, VolumeDim, Frame>>>(
        db::item_type<Tags::IndicesOfFilledInterpPoints>{},
        db::item_type<Tags::TemporalIds<Metavariables>>{}, std::move(domain),
        db::item_type<::Tags::Variables<typename InterpolationTargetTag::
                                            vars_to_interpolate_to_target>>{}));
  }
};

}  // namespace Actions
}  // namespace intrp
