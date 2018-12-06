// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines action AdvanceTime

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace OptionTags {
struct TimeStepper;
}  // namespace OptionTags
namespace Tags {
template <typename Tag>
struct Next;
struct TimeId;
struct TimeStep;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// \brief Advance time one substep
///
/// Uses:
/// - ConstGlobalCache: OptionTags::TimeStepper
/// - DataBox: Tags::TimeId, Tags::TimeStep
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: Tags::TimeId, Tags::TimeStep
struct AdvanceTime {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<Tags::TimeId, Tags::Next<Tags::TimeId>, Tags::TimeStep>(
        make_not_null(&box), [&cache](const gsl::not_null<TimeId*> time_id,
                                      const gsl::not_null<TimeId*> next_time_id,
                                      const gsl::not_null<TimeDelta*>
                                          time_step) noexcept {
          const auto& time_stepper =
              Parallel::get<OptionTags::TimeStepper>(cache);
          *time_id = *next_time_id;
          *time_step = time_step->with_slab(time_id->time().slab());
          *next_time_id = time_stepper.next_time_id(*next_time_id, *time_step);
        });

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
