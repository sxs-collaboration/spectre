// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "NumericalAlgorithms/Interpolation/PointInfoTag.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace intrp {
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Adds interpolation point holders to the Element's DataBox.
///
/// This action is for the case in which the points are time-independent.
///
/// This action should be placed in the Initialization PDAL for DgElementArray.
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds:
///   - `intrp::Tags::InterpPointInfo<Metavariables>`
/// - Removes: nothing
/// - Modifies: nothing
struct ElementInitInterpPoints {
  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename Metavariables,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // It appears that clang-tidy is unhappy with 'if constexpr',
    // hence the directives below.
    if constexpr (tmpl::list_contains_v<
                      // NOLINTNEXTLINE clang-tidy wants extra braces.
                      DbTags, intrp::Tags::InterpPointInfo<Metavariables>>) {
      ERROR(
          "Found 'intrp::Tags::InterpPointInfo<Metavariables>' in DataBox, but "
          "it should not be there because ElementInitInterpPoints adds it.");
      return std::forward_as_tuple(std::move(box));
    } else {  // NOLINT clang-tidy thinks 'if' and 'else' not indented the same
      using point_info_type = tuples::tagged_tuple_from_typelist<
          db::wrap_tags_in<Tags::point_info_detail::WrappedPointInfoTag,
                           typename Metavariables::interpolation_target_tags,
                           Metavariables>>;
      return std::make_tuple(
          db::create_from<
              db::RemoveTags<>,
              db::AddSimpleTags<intrp::Tags::InterpPointInfo<Metavariables>>>(
              std::move(box), point_info_type{}));
    }
  }
};
}  // namespace Actions
}  // namespace intrp
