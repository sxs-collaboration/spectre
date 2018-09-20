// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions for tracing Actions

#pragma once

#ifdef SPECTRE_CHARM_PROJECTIONS

#include <charm++.h>

//#include "Utilities/Papi.hpp"

//#include "Parallel/Algorithm.hpp"
#include "Parallel/Info.hpp"
#include "Utilities/TMPL.hpp"

using the_trace_actions_list = actions_list;

namespace detail {
// @{
/*!
 * \ingroup Profiling
 * \brief Called after an Action to record the time it took to execute and any
 * PAPI counters
 *
 * A Charm++ bracketed user event is recorded for the particular Action given
 * the `start_time`. The `start_time` should be the return value of
 * `start_trace_action` called before the Action executed. If any PAPI counters
 * should be recorded then the counters are stopped, causing an ERROR if there
 * are problems stopping the counters. The values of the counters are read into
 * `read_events` and then recorded into Charm++ user stats (supported from
 * Charm++ v6.8.0 and newer) for analysis.
 */

template <
    typename Action,
    Requires<tmpl::found<the_trace_actions_list,
                         tmpl::bind<std::is_same, tmpl::_1, Action>> =
                 nullptr> void stop_trace_action(const double start_time) {
  static_assert(tmpl::size<the_trace_actions_list>::value < 1000,
                "Cannot have more than 1000 Actions to trace");
  traceUserBracketEvent(tmpl::index_of<the_trace_actions_list, Action>::value,
                        start_time, Parallel::wall_time());
#ifdef SPECTRE_PAPI_COUNTERS
  const auto action_number =
      tmpl::index_of<the_trace_actions_list, Action>::value;
  std::array<long long, papi_event_names.size()> read_events;
  long long ret = PAPI_OK;
  if ((ret = PAPI_stop_counters(read_events.data(), read_events.size())) !=
      PAPI_OK) {
    ERROR("PAPI failed to stop and read counts: " << PAPI_strerror(ret));
  }
  for (size_t i = 0; i < read_events.size(); ++i) {
    updateStat(
        charm_papi_events_offset + i + action_number * read_events.size(),
        read_events[i]);
  }
#endif  // defined(SPECTRE_PAPI_COUNTERS)
}

/// \cond HIDDEN_SYMBOLS
template <
    typename Action,
    Requires<not tmpl::found<the_trace_actions_list,
                             tmpl::bind<std::is_same, tmpl::_1, Action>>> =
        nullptr>
constexpr void stop_trace_action(const double /*start_time*/) {}
/// \endcond
// @}

/*!
 * \ingroup Profiling
 * \brief Start tracing an Action, returns the walltime of the call
 *
 * Checks if the Action `Action` is in the `ActionList` and
 * if so attempts to start PAPI counters if requested. An error occurs if the
 * counters were unable to start for any reason, included is an error message.
 * Typically the `Invalid arguments` means either one of the `papi_event_ids` is
 * not an available hardware PMU, or the counters were already started once and
 * not stopped before this call.
 */
template <typename Action>
double start_trace_action() {
#ifdef SPECTRE_PAPI_COUNTERS
  if (tmpl::found<the_trace_actions_list,
                  tmpl::bind<std::is_same, Action, tmpl::_1>>::value) {
    long long ret = PAPI_OK;
    if (UNLIKELY((ret = PAPI_start_counters(
                      papi_event_ids.data(),
                      static_cast<int>(papi_event_ids.size()))) != PAPI_OK)) {
      ERROR("PAPI failed to start counters: "
            << PAPI_strerror(ret) << "\npapi_event_ids: " << papi_event_ids);
    }
  }
#endif  // defined(SPECTRE_PAPI_COUNTERS)
  return Parallel::wall_time();
}
}  // namespace detail
#endif  // defined(SPECTRE_CHARM_PROJECTIONS)
