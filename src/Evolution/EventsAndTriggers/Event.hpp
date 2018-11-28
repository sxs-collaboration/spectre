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
namespace Events {
template <typename EventRegistrars>
class Completion;
}  // namespace Events
/// \endcond

/// \ingroup EventsAndTriggersGroup
namespace Events {
/// Registrars for Events
namespace Registrars {}
}  // namespace Events

/// \ingroup EventsAndTriggersGroup
/// Base class for something that can happen during a simulation (such
/// as an observation).
template <typename EventRegistrars>
class Event : public PUP::able {
 protected:
  /// \cond
  Event() = default;
  Event(const Event&) = default;
  Event(Event&&) = default;
  Event& operator=(const Event&) = default;
  Event& operator=(Event&&) = default;
  /// \endcond

 public:
  ~Event() override = default;

  WRAPPED_PUPable_abstract(Event);  // NOLINT

  using default_events = tmpl::list<Events::Completion<EventRegistrars>>;

  using creatable_classes =
      tmpl::append<default_events, Registration::registrants<EventRegistrars>>;

  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename ComponentPointer>
  void run(const db::DataBox<DbTags>& box,
           Parallel::ConstGlobalCache<Metavariables>& cache,
           const ArrayIndex& array_index,
           const ComponentPointer /*meta*/) noexcept {
    call_with_dynamic_type<void, creatable_classes>(
        this,
        [&box, &cache, &array_index](auto* const event) noexcept {
          using EventType = std::decay_t<decltype(*event)>;
          db::apply<typename EventType::argument_tags>(
              *event, box, cache, array_index, ComponentPointer{});
        });
  }
};

#include "Evolution/EventsAndTriggers/Completion.hpp"
