// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines tags related to Events and Triggers

#pragma once

#include "Evolution/EventsAndTriggers/EventsAndTriggers.hpp"
#include "Options/Options.hpp"

namespace Tags {
/// \cond
struct EventsAndTriggersTagBase : db::BaseTag {};
/// \endcond

/// The `OptionTags::EventsAndTriggers` in the DataBox
template <typename EventRegistrars, typename TriggerRegistrars>
struct EventsAndTriggers : EventsAndTriggersTagBase, db::SimpleTag {
  static std::string name() noexcept { return "EventsAndTriggers"; }
  using type = ::EventsAndTriggers<EventRegistrars, TriggerRegistrars>;
};
}  // namespace Tags

namespace OptionTags {
/// \cond
struct EventsAndTriggersTagBase {};
/// \endcond

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
struct EventsAndTriggers : EventsAndTriggersTagBase {
  using type = ::EventsAndTriggers<EventRegistrars, TriggerRegistrars>;
  static constexpr OptionString help = "Events to run at triggers";
  using container_tag =
      Tags::EventsAndTriggers<EventRegistrars, TriggerRegistrars>;
};
}  // namespace OptionTags
