// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Inboxes.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Tags.hpp"
#include "Time/Actions/ChangeSlabSize.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::Worldtube::Actions {

/*!
 * \brief Waits for the data from all neighboring elements and changes the slab
 * size if a change in the global time step is detected.
 * \details We check the slab size of the time step id sent by the elements. If
 * this is different from the slab size currently used by the worldtube
 * singleton, we assume a global slab size change has occurred in the elements
 * and adjust the worldtube slab size accordingly.
 */
struct ChangeSlabSize {
  static constexpr size_t Dim = 3;
  using inbox_tags = tmpl::list<
      ::CurvedScalarWave::Worldtube::Tags::SphericalHarmonicsInbox<Dim>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& time_step_id = db::get<::Tags::TimeStepId>(box);
    const auto& inbox =
        tuples::get<Tags::SphericalHarmonicsInbox<Dim>>(inboxes);
    if (inbox.empty()) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    ASSERT(inbox.size() == 1,
           "Received data from two different time step ids.");
    const auto inbox_time_step_id = inbox.begin()->first;
    const auto& inbox_slab = inbox_time_step_id.step_time().slab();
    // we received data from a time step id with a different slab size,
    // indicating a change in the global time step.
    if (inbox_time_step_id.step_time().slab() !=
        time_step_id.step_time().slab()) {
      ASSERT(inbox_slab.start() == time_step_id.step_time().slab().start(),
             "The new slab should start at the same time as the old one.");
      // time_step_id comparison does NOT compare slabs.
      ASSERT(
          inbox_time_step_id == time_step_id,
          "Detected change in global time step id but the new time step id is "
          "different.");
      const auto new_step = inbox_slab.duration();
      const auto new_next_time_step_id =
          db::get<::Tags::TimeStepper<>>(box).next_time_id(inbox_time_step_id,
                                                           new_step);
      db::mutate<::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
                 ::Tags::Next<::Tags::TimeStep>, ::Tags::TimeStepId>(
          [&new_next_time_step_id, &new_step, &inbox_time_step_id](
              const gsl::not_null<TimeStepId*> next_time_step_id,
              const gsl::not_null<TimeDelta*> time_step,
              const gsl::not_null<TimeDelta*> next_time_step,
              const gsl::not_null<TimeStepId*> local_time_step_id) {
            *next_time_step_id = new_next_time_step_id;
            *time_step = new_step;
            *next_time_step = new_step;
            *local_time_step_id = inbox_time_step_id;
          },
          make_not_null(&box));
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace CurvedScalarWave::Worldtube::Actions
