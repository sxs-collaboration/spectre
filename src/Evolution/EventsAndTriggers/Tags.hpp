// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines tags related to Events and Triggers

#pragma once

#include "Evolution/EventsAndTriggers/EventsAndTriggers.hpp"
#include "Options/Options.hpp"

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
};
}  // namespace OptionTags
