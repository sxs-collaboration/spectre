// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
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
/// - GlobalCache:
///   - Metavariables::variable_fixer
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: Metavariables::variable_fixer::return_tags
template <typename VariableFixer>
struct FixVariables {
  using const_global_cache_tags =
      tmpl::list<Tags::VariableFixer<VariableFixer>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::size<DbTagsList>::value != 0> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& variable_fixer = get<Tags::VariableFixer<VariableFixer>>(cache);
    db::mutate_apply(variable_fixer, make_not_null(&box));
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace VariableFixing
