// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines tags related to Events and Triggers

#pragma once

#include <vector>
#include <memory>

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup EventsAndTriggersGroup
/// Contains the events and triggers
///
/// In yaml this is specified as a map of triggers to lists of events:
/// \code{.yaml}
/// EventsAndTriggers:
///   ? TriggerA:
///       OptionsForTriggerA
///   : - Event1:
///         OptionsForEvent1
///     - Event2:
///         OptionsForEvent2
///   ? TriggerB:
///       OptionsForTriggerB
///   : - Event3:
///         OptionsForEvent3
///     - Event4:
///         OptionsForEvent4
/// \endcode
struct EventsAndTriggers {
  using type = ::EventsAndTriggers;
  static constexpr Options::String help = "Events to run at triggers";
  // When the template arguments to this struct are sufficiently
  // complicated, pretty_type::short_name() run on this struct returns
  // something that is neither pretty nor short, and leads to an
  // OptionParser run-time error saying that an option name is greater
  // than 21 characters.  Adding the name() function below bypasses
  // pretty_type::short_name().
  static std::string name() { return "EventsAndTriggers"; }
};

/// \brief A list of events to run at cleanpppp.
///
/// See `Actions::RunEventsOnFailure` for details and caveats.
struct EventsRunAtCleanup {
  using type = std::vector<std::unique_ptr<::Event>>;
  static constexpr Options::String help =
      "Events to run during the cleanup phase. This is generally intended for "
      "dumping volume data to diagnose failure reasons.";
};
}  // namespace OptionTags

namespace Tags {
/// \ingroup EventsAndTriggersGroup
/// Contains the events and triggers
struct EventsAndTriggers : db::SimpleTag {
  using type = ::EventsAndTriggers;
  using option_tags = tmpl::list<::OptionTags::EventsAndTriggers>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& events_and_triggers) {
    return deserialize<type>(serialize<type>(events_and_triggers).data());
  }
};

/// \brief Events to be run on elements during the
/// `Parallel::Phase::PostFailureCleanup` phase.
///
/// Useful for troubleshooting runs that are failing.
struct EventsRunAtCleanup : db::SimpleTag {
  using type = std::vector<std::unique_ptr<::Event>>;
  using option_tags = tmpl::list<OptionTags::EventsRunAtCleanup>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& events_run_at_cleanup) {
    return deserialize<type>(serialize<type>(events_run_at_cleanup).data());
  }
};
}  // namespace Tags
