// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup EventsAndTriggersGroup
namespace Triggers {
template <typename>
class Always;
template <typename>
class And;
template <typename>
class Not;
template <typename>
class Or;
}  // namespace Triggers

namespace Trigger_detail {
template <typename KnownTriggers>
using logical_triggers =
    tmpl::list<Triggers::Always<KnownTriggers>, Triggers::And<KnownTriggers>,
               Triggers::Not<KnownTriggers>, Triggers::Or<KnownTriggers>>;
}  // namespace Trigger_detail

/// \ingroup EventsAndTriggersGroup
/// Base class for checking whether to run an Event.
template <typename KnownTriggers>
class Trigger : public PUP::able {
 protected:
  /// \cond
  Trigger() = default;
  Trigger(const Trigger&) = default;
  Trigger(Trigger&&) = default;
  Trigger& operator=(const Trigger&) = default;
  Trigger& operator=(Trigger&&) = default;
  /// \endcond

 public:
  ~Trigger() override = default;

  WRAPPED_PUPable_abstract(Trigger);  // NOLINT

  using creatable_classes = tmpl::remove_duplicates<tmpl::append<
      Trigger_detail::logical_triggers<KnownTriggers>,
      typename KnownTriggers::template type<KnownTriggers>>>;

  template <typename DbTags>
  bool is_triggered(const db::DataBox<DbTags>& box) noexcept {
    return call_with_dynamic_type<bool, creatable_classes>(
        this,
        [&box](auto* const trigger) noexcept {
          using TriggerType = std::decay_t<decltype(*trigger)>;
          return db::apply<typename TriggerType::argument_tags>(
              *trigger, box);
        });
  }
};

#include "Evolution/EventsAndTriggers/LogicalTriggers.hpp"
