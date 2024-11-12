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
#include "ParallelAlgorithms/EventsAndTriggers/WhenToCheck.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup EventsAndTriggersGroup
/// Contains the events and triggers
///
/// In yaml this is specified as a map of triggers to lists of events:
/// \code{.yaml}
/// EventsAndTriggersAtSlabs:
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
template <Triggers::WhenToCheck WhenToCheck>
struct EventsAndTriggers {
  using type = ::EventsAndTriggers;
  static constexpr Options::String help = "Events to run at triggers";
  // When the template arguments to this struct are sufficiently
  // complicated, pretty_type::short_name() run on this struct returns
  // something that is neither pretty nor short, and leads to an
  // OptionParser run-time error saying that an option name is greater
  // than 21 characters.  Adding the name() function below bypasses
  // pretty_type::short_name().
  static std::string name() {
    return "EventsAndTriggers" + get_output(WhenToCheck);
  }
};

namespace EventsRunAtCleanup {
struct Group {
  static std::string name() { return "EventsRunAtCleanup"; }
  static constexpr Options::String help =
      "Options related to running events on failure.  This is generally "
      "intended for dumping volume data to diagnose failure reasons.";
};

/// \brief A list of events to run at cleanup.
///
/// See `Actions::RunEventsOnFailure` for details and caveats.
struct Events {
  static std::string name() { return "Events"; }
  using type = std::vector<std::unique_ptr<::Event>>;
  static constexpr Options::String help =
      "Events to run during the cleanup phase.";
  using group = Group;
};

/// \brief Observation value for Actions::RunEventsOnFailure.
struct ObservationValue {
  using type = double;
  static constexpr Options::String help =
      "Observation value for events run during the cleanup phase.";
  using group = Group;
};
}  // namespace EventsRunAtCleanup
}  // namespace OptionTags

namespace Tags {
/// \ingroup EventsAndTriggersGroup
/// Contains the events and triggers
template <Triggers::WhenToCheck WhenToCheck>
struct EventsAndTriggers : db::SimpleTag {
  using type = ::EventsAndTriggers;
  using option_tags = tmpl::list<::OptionTags::EventsAndTriggers<WhenToCheck>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& events_and_triggers) {
    return deserialize<type>(serialize<type>(events_and_triggers).data());
  }
  static std::string name() {
    return "EventsAndTriggers" + get_output(WhenToCheck);
  }
};

/// \brief Events to be run on elements during the
/// `Parallel::Phase::PostFailureCleanup` phase.
///
/// Useful for troubleshooting runs that are failing.
struct EventsRunAtCleanup : db::SimpleTag {
  using type = std::vector<std::unique_ptr<::Event>>;
  using option_tags = tmpl::list<OptionTags::EventsRunAtCleanup::Events>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& events_run_at_cleanup) {
    return deserialize<type>(serialize<type>(events_run_at_cleanup).data());
  }
};

/// \brief Observation value for Actions::RunEventsOnFailure.
struct EventsRunAtCleanupObservationValue : db::SimpleTag {
  using type = double;
  using option_tags =
      tmpl::list<OptionTags::EventsRunAtCleanup::ObservationValue>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& value) { return value; }
};
}  // namespace Tags
