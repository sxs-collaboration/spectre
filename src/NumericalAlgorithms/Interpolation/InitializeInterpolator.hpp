// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/Tags.hpp" // IWYU pragma: keep
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace intrp {
namespace Tags {
struct NumberOfElements;
template <typename Metavariables>
struct InterpolatedVarsHolders;
template <typename Metavariables>
struct VolumeVarsInfo;
}  // namespace Tags
}  // namespace intrp
/// \endcond

namespace intrp {
namespace Actions {

/// \ingroup ActionsGroup
/// \brief Initializes an Interpolator
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds:
///   - `Tags::NumberOfElements`
///   - `Tags::VolumeVarsInfo<Metavariables>`
///   - `Tags::InterpolatedVarsHolders<Metavariables>`
/// - Removes: nothing
/// - Modifies: nothing
struct InitializeInterpolator {
  template <typename Metavariables>
  using return_tag_list =
      tmpl::list<Tags::NumberOfElements,
                 Tags::VolumeVarsInfo<Metavariables>,
                 Tags::InterpolatedVarsHolders<Metavariables>>;

  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<not tmpl::list_contains_v<DbTagsList, Tags::NumberOfElements> and
               not tmpl::list_contains_v<
                   DbTagsList, Tags::VolumeVarsInfo<Metavariables>> and
               not tmpl::list_contains_v<
                   DbTagsList, Tags::InterpolatedVarsHolders<Metavariables>>> =
          nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        db::create_from<db::RemoveTags<>,
                        db::get_items<return_tag_list<Metavariables>>>(
            std::move(box), 0_st,
            db::item_type<Tags::VolumeVarsInfo<Metavariables>>{},
            db::item_type<Tags::InterpolatedVarsHolders<Metavariables>>{}));
  }

  template <
      typename DbTagsList, typename... InboxTags, typename Metavariables,
      typename ArrayIndex, typename ActionList, typename ParallelComponent,
      Requires<tmpl::list_contains_v<DbTagsList, Tags::NumberOfElements> and
               tmpl::list_contains_v<DbTagsList,
                                     Tags::VolumeVarsInfo<Metavariables>> and
               tmpl::list_contains_v<DbTagsList, Tags::InterpolatedVarsHolders<
                                                     Metavariables>>> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    return {std::move(box)};
  }
};

}  // namespace Actions
}  // namespace intrp
