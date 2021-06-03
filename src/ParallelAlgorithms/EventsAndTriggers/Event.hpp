// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup EventsAndTriggersGroup
namespace Events {}

/// \ingroup EventsAndTriggersGroup
/// Base class for something that can happen during a simulation (such
/// as an observation).
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
  explicit Event(CkMigrateMessage* msg) noexcept : PUP::able(msg) {}

  WRAPPED_PUPable_abstract(Event);  // NOLINT

  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename ComponentPointer>
  void run(const db::DataBox<DbTags>& box,
           Parallel::GlobalCache<Metavariables>& cache,
           const ArrayIndex& array_index,
           const ComponentPointer /*meta*/) const noexcept {
    using factory_classes =
        typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            box))>::factory_creation::factory_classes;
    call_with_dynamic_type<void, tmpl::at<factory_classes, Event>>(
        this, [&box, &cache, &array_index](auto* const event) noexcept {
          db::apply(*event, box, cache, array_index, ComponentPointer{});
        });
  }

  /// Whether the event uses anything depending on the
  /// evolved_variables.  If this returns false, anything depending on
  /// the evolved variables may have an incorrect value when the event
  /// is run.
  virtual bool needs_evolved_variables() const noexcept = 0;
};
