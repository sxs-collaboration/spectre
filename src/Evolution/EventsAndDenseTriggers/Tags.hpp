// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "Options/Options.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution {
namespace OptionTags {
/*!
 * \ingroup EventsAndTriggersGroup
 * \brief The Event%s to run based on DenseTrigger%s, similar to
 * OptionTags::EventsAndTriggers
 */
struct EventsAndDenseTriggers {
  using type = std::vector<evolution::DenseTriggerAndEventsConstruction>;
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
  static type create_from_options(
      const std::vector<evolution::DenseTriggerAndEventsConstruction>&
          construction) {
    // The deserialize(serialize()) is to deal with unique_ptrs in the data
    // structures since we don't have `get_clone` implemented for
    // triggers/events
    return deserialize<type>(
        serialize(
            type{deserialize<
                std::vector<evolution::DenseTriggerAndEventsConstruction>>(
                serialize(construction).data())})
            .data());
  }
};

/*!
 * \brief Vector of DenseTrigger%s and Event%s used to construct
 * `evolution::Tags::EventsAndDenseTriggers`.
 *
 * \warning Upon option construction, this tag will NOT necessarily contain all
 * of the DenseTrigger%s and Event%s that will be used in the evolution. Some
 * may be added during other `Parallel::Phase`s of the execution using the
 * `evolution::EventsAndDenseTrigger::add_trigger_and_events` method. Therefore,
 * this tag should likely be placed in the `mutable_global_cache_tags` rather
 * than the `const_global_cache_tags`.
 */
struct EventsAndDenseTriggersOptions : db::SimpleTag {
  using type = std::vector<evolution::DenseTriggerAndEventsConstruction>;
  using option_tags = tmpl::list<OptionTags::EventsAndDenseTriggers>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) {
    // The deserialize(serialize()) is to deal with unique_ptrs in the data
    // structures since we don't have `get_clone` implemented for
    // triggers/events
    return deserialize<
        std::vector<evolution::DenseTriggerAndEventsConstruction>>(
        serialize(option).data());
  }
};
}  // namespace Tags
}  // namespace evolution
