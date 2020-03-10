// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines tags related to Events and Triggers

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Options/Options.hpp"
#include "Parallel/Serialize.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/EventsAndTriggers.hpp"

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
template <typename EventRegistrars, typename TriggerRegistrars>
struct EventsAndTriggers {
  using type = ::EventsAndTriggers<EventRegistrars, TriggerRegistrars>;
  static constexpr OptionString help = "Events to run at triggers";
  // When the template arguments to this struct are sufficiently
  // complicated, pretty_type::short_name() run on this struct returns
  // something that is neither pretty nor short, and leads to an
  // OptionParser run-time error saying that an option name is greater
  // than 21 characters.  Adding the name() function below bypasses
  // pretty_type::short_name().
  static std::string name() noexcept { return "EventsAndTriggers"; }
};
}  // namespace OptionTags

namespace Tags {
/// \cond
struct EventsAndTriggersBase : db::BaseTag {};
/// \endcond

/// \ingroup EventsAndTriggersGroup
/// Contains the events and triggers
template <typename EventRegistrars, typename TriggerRegistrars>
struct EventsAndTriggers : EventsAndTriggersBase, db::SimpleTag {
  using type = ::EventsAndTriggers<EventRegistrars, TriggerRegistrars>;
  using option_tags = tmpl::list<
      ::OptionTags::EventsAndTriggers<EventRegistrars, TriggerRegistrars>>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& events_and_triggers) noexcept {
    return deserialize<type>(serialize<type>(events_and_triggers).data());
  }
};
}  // namespace Tags
