// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "Parallel/AlgorithmExecution.hpp"

/// \cond
namespace tuples {
template <typename... InboxTags>
struct TaggedTuple;
}  // namespace tuples
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Parallel {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Terminate the algorithm to proceed to the next phase.
 */
struct TerminatePhase {
  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      DataBox& /*box*/, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ActionList /*meta*/,
      // NOLINTNEXTLINE(readability-avoid-const-params-in-decls)
      const ParallelComponent* const /*meta*/) {
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

}  // namespace Actions
}  // namespace Parallel
