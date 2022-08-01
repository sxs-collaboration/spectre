// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace Actions {
/// \ingroup ActionsGroup
/// \brief Compute the conservative variables from the primitive variables
///
/// Uses:
/// - DataBox: Items in system::conservative_from_primitive::argument_tags
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: Metavariables::system::conservative_from_primitive::return_tags
struct UpdateConservatives {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::size<DbTagsList>::value != 0> = nullptr>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate_apply<
        typename Metavariables::system::conservative_from_primitive>(
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
