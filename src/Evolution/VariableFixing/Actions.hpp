// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace VariableFixing {
namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup VariableFixingGroup
/// \brief Adjust variables with a variable fixer.
///
/// Typically this action is called to adjust either conservative or primitive
/// variables when they violate physical constraints.  See the individual
/// variable fixers in the VariableFixing namespace for more details.
///
/// Uses:
/// - DataBox:
///   - Metavariables::variable_fixer::argument_tags
/// - ConstGlobalCache:
///   - Metavariables::variable_fixer
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: Metavariables::variable_fixer::return_tags
template <typename Metavariables>
struct FixVariables {
  using const_global_cache_tags =
      tmpl::list<typename Metavariables::variable_fixer>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            Requires<tmpl::size<DbTagsList>::value != 0> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& variable_fixer =
        get<typename Metavariables::variable_fixer>(cache);
    db::mutate_apply<
        typename Metavariables::variable_fixer::type::return_tags,
        typename Metavariables::variable_fixer::type::argument_tags>(
        variable_fixer, make_not_null(&box));
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace VariableFixing
