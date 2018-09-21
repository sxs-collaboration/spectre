// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions to register Actions for tracing

#pragma once

#include <charm++.h>
#include <cstring>

#include "Parallel/CharmRegistration.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
#ifdef SPECTRE_CHARM_PROJECTIONS
// Nothing to do for empty ActionList
template <typename ActionList, typename TotalSize = tmpl::size<ActionList>,
          Requires<(tmpl::size<ActionList>::value == 0)> = nullptr>
constexpr SPECTRE_ALWAYS_INLINE void register_charm_trace_user_event() {}

// Non-Empty ActionList
// Register events recursively
template <typename ActionList, typename TotalSize = tmpl::size<ActionList>,
          Requires<(tmpl::size<ActionList>::value > 0)> = nullptr>
void register_charm_trace_user_event() {
  const auto action_number = TotalSize::value - tmpl::size<ActionList>::value;

  // note: eventstr is not temporary and not freed on purpose, because we need
  // either a constant string or a dynamically allocated string that is NOT
  // freed by the program
  std::string* eventstr =
      new std::string(Parallel::charmxx::get_template_parameters_as_string<
                      tmpl::front<ActionList>>());
  traceRegisterUserEvent(eventstr->c_str(), action_number);
#ifdef SPECTRE_PAPI_COUNTERS
  for (size_t i = 0; i < papi_event_names.size(); ++i) {
    // Note: the pointer char* named "buffer" is freed by Charm++ internally
    char* buffer =
        new char[std::strlen(papi_event_names[i]) +
                 std::strlen(tmpl::front<ActionList>::event_name) + 3];
    // Stats are named "Action::event_name + '_' + papi_event_names"
    std::strcpy(buffer, tmpl::front<ActionList>::event_name);
    buffer[std::strlen(tmpl::front<ActionList>::event_name)] = '_';
    std::strcpy(&buffer[std::strlen(tmpl::front<ActionList>::event_name)] + 1,
                papi_event_names[i]);
    // charm_papi_events_offset is defined in Utilities/Papi.hpp and is what
    // will be an inline variable in C++17 (it has no external linkage). It is
    // used to ensure an offset from zero of the user stat options.
    traceRegisterUserStat(buffer, charm_papi_events_offset + i +
                                      action_number * papi_event_names.size());
  }
#endif  // defined(SPECTRE_PAPI_COUNTERS)
  register_charm_trace_user_event<tmpl::pop_front<ActionList>, TotalSize>();
}

// User stats are not supported for Charm++ versions before v6.8.0, so we check
// if the function exists, otherwise don't add them
#ifdef SPECTRE_PROJECTIONS_USER_STATS
void register_specified_user_stats() {
  for (size_t i = 0; i < user_stat_names.size(); ++i) {
    traceRegisterUserStat(user_stat_names[i], projections_user_stat_offset + i);
  }
}
#else   // defined(SPECTRE_PROJECTIONS_USER_STATS)
constexpr SPECTRE_ALWAYS_INLINE void register_specified_user_stats() {}
#endif  // defined(SPECTRE_PROJECTIONS_USER_STATS)

template <typename ActionList>
inline void register_events_to_trace() {
  // PAPI not yet ported, hence commented out
  // init_papi_for_parallel Parallel::init_papi_for_parallel();
  Parallel::register_charm_trace_user_event<ActionList>();
  traceRegisterUserEvent("Non-action time",
                         SPECTRE_CHARM_NON_ACTION_WALLTIME_EVENT_ID);
#ifdef SPECTRE_CHARM_RECEIVE_MAP_DATA_EVENT_ID
  traceRegisterUserEvent("Receive Map Data",
                         SPECTRE_CHARM_RECEIVE_MAP_DATA_EVENT_ID);
#endif  // defined(SPECTRE_CHARM_RECEIVE_MAP_DATA_EVENT_ID)
  // Register additional user-specified stats
  register_specified_user_stats();
}
#else   // defined(SPECTRE_CHARM_PROJECTIONS)
constexpr SPECTRE_ALWAYS_INLINE void register_events_to_trace() {}
#endif  // defined(SPECTRE_CHARM_PROJECTIONS)
}  // namespace Parallel
