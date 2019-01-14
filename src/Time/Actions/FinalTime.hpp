// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines action FinalTime

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Time/Time.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace OptionTags {
struct FinalTime;
}  // namespace OptionTags
namespace Tags {
struct Time;
struct TimeStep;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// \brief Terminate after reaching a specified time
///
/// Uses:
/// - ConstGlobalCache: OptionTags::FinalTime
/// - DataBox: Tags::Time, Tags::TimeStep
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: nothing
struct FinalTime {
  using const_global_cache_tags = tmpl::list<OptionTags::FinalTime>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&, bool> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const double final_time = Parallel::get<OptionTags::FinalTime>(cache);
    const Time& time = db::get<Tags::Time>(box);
    const TimeDelta& time_step = db::get<Tags::TimeStep>(box);

    return {std::move(box), time_step.is_positive()
                                ? time.value() >= final_time
                                : time.value() <= final_time};
  }
};
}  // namespace Actions
