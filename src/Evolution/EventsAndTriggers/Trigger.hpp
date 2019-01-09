// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Triggers {
template <typename TriggerRegistrars>
class Always;
template <typename TriggerRegistrars>
class And;
template <typename TriggerRegistrars>
class Not;
template <typename TriggerRegistrars>
class Or;
}  // namespace Triggers
/// \endcond

/// \ingroup EventsAndTriggersGroup
namespace Triggers {
/// Registrars for Triggers
namespace Registrars {}
}  // namespace Triggers

/// \ingroup EventsAndTriggersGroup
/// Base class for checking whether to run an Event.
template <typename TriggerRegistrars>
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

  using default_triggers = tmpl::list<
      Triggers::Always<TriggerRegistrars>, Triggers::And<TriggerRegistrars>,
      Triggers::Not<TriggerRegistrars>, Triggers::Or<TriggerRegistrars>>;

  using creatable_classes =
      tmpl::append<default_triggers,
                   Registration::registrants<TriggerRegistrars>>;

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
