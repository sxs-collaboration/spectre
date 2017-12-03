// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines action UpdateU

#pragma once

#include <tuple>
#include <utility>

#include "DataStructures/DataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// \brief Perform variable updates for one substep
///
/// Uses:
/// - ConstGlobalCache: CacheTags::TimeStepper
/// - DataBox: system::variables_tag, system::dt_variables_tag,
///   Tags::HistoryEvolvedVariables<system::variables_tag,
///                                 system::dt_variables_tag>,
///   Tags::Time, Tags::TimeStep
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: system::dt_variables_tag
/// - Modifies: system::variables_tag,
///   Tags::HistoryEvolvedVariables<system::variables_tag,
///                                 system::dt_variables_tag>
struct UpdateU {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using variables_tag = typename Metavariables::system::variables_tag;
    using dt_variables_tag = typename Metavariables::system::dt_variables_tag;

    db::mutate<variables_tag, dt_variables_tag,
               Tags::HistoryEvolvedVariables<variables_tag, dt_variables_tag>>(
        box, [&cache](auto& vars, auto& dt_vars, auto& history,
                      const auto& time, const auto& time_step) noexcept {
          const auto& time_stepper =
              Parallel::get<CacheTags::TimeStepper>(cache);

          history.emplace_back(time, vars, std::move(dt_vars));
          time_stepper.update_u(make_not_null(&vars), history, time_step);
          history.erase(history.begin(), time_stepper.needed_history(history));
        },
        db::get<Tags::Time>(box),
        db::get<Tags::TimeStep>(box));

    return std::make_tuple(
        db::create_from<db::RemoveTags<dt_variables_tag>, db::AddTags<>>(
            std::move(box)));
  }
};
}  // namespace Actions
