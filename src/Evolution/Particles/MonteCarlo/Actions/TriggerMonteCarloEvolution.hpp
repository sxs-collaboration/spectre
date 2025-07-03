// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Particles/MonteCarlo/Actions/Labels.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "ParallelAlgorithms/Actions/Goto.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace Particles::MonteCarlo::Actions {
/*!
 * \brief Goes to `Labels::BeginMonteCarlo` or `Labels::EndMonteCarlo` depending
 * on whether we are at the end of a full time step or at an intermediate step
 * of the timestepping algorithm.
 *
 * GlobalCache: nothing
 *
 * DataBox:
 * - Uses:
 *   - Tags::Next<::Tags::TimeStepId>
 */
struct TriggerMonteCarloEvolution {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& next_time_id = db::get<::Tags::Next<::Tags::TimeStepId>>(box);
    // We only run MC if we are at the beginning of a full time step
    const bool trigger_mc = (next_time_id.substep() == 0);
    if (trigger_mc) {
      // Note: we jump to the `Label+1` because the label actions don't do
      // anything anyway
      const size_t mc_index =
          tmpl::index_of<ActionList,
                         ::Actions::Label<Labels::BeginMonteCarlo>>::value +
          1;
      return {Parallel::AlgorithmExecution::Continue, mc_index};
    } else {
      // Here we use `Label', because there might not be a `Label + 1'
      const size_t post_mc_index =
          tmpl::index_of<ActionList,
                         ::Actions::Label<Labels::EndMonteCarlo>>::value;
      return {Parallel::AlgorithmExecution::Continue, post_mc_index};
    }
  }
};
}  // namespace Particles::MonteCarlo::Actions
