// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "Parallel/AlgorithmExecution.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Time/Tags/MinimumTimeStep.hpp"
#include "Time/Tags/StepChoosers.hpp"
#include "Time/TakeStep.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
struct AllStepChoosers;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// Record time stepper data, adjust the step size, and take a step.
/// Usually this is done as part of the
/// evolution::dg::Actions::ComputeTimeDerivative action.  This is for
/// executables that do not use that.
template <typename System, typename StepChoosersToUse = AllStepChoosers>
struct TakeLtsStep {
  using const_global_cache_tags = tmpl::list<::Tags::MinimumTimeStep>;
  using simple_tags = typename ::Actions::UpdateU<System, true>::simple_tags;
  using compute_tags = typename ::Actions::UpdateU<System, true>::compute_tags;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    take_step<System, true, StepChoosersToUse>(make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
