// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "Options/Options.hpp"
#include "Parallel/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution {
namespace OptionTags {
/*!
 * \ingroup EventsAndTriggersGroup
 * \brief The Event%s to run based on DenseTrigger%s, similar to
 * OptionTags::EventsAndTriggers
 */
struct EventsAndDenseTriggers {
  using type = evolution::EventsAndDenseTriggers;
  static constexpr Options::String help = "Events to run at arbitrary times";
};
}  // namespace OptionTags

namespace Tags {
/*!
 * \ingroup EventsAndTriggersGroup
 * \brief The Event%s to run based on DenseTrigger%s
 */
struct EventsAndDenseTriggers : db::SimpleTag {
  using type = evolution::EventsAndDenseTriggers;
  using option_tags = tmpl::list<OptionTags::EventsAndDenseTriggers>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& events_and_triggers) noexcept {
    return deserialize<type>(serialize(events_and_triggers).data());
  }
};
}  // namespace Tags
}  // namespace evolution
