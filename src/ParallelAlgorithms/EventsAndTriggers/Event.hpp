// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/ObservationBox.hpp"
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
///
/// Derived events must have a `compute_tags_for_observation_box` that is a
/// `tmpl::list` of simple or compute tags. Simple tags are assumed to already
/// be in the `DataBox`. Evolved variables, for example, would be listed as
/// simple tags. The compute tags are used to compute additional quantities that
/// may be observed. For example, in the scalar wave system the 1- and 2-index
/// constraints would be added as compute tags, as well as anything they depend
/// on that's not already in the `DataBox`.
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
  explicit Event(CkMigrateMessage* msg) : PUP::able(msg) {}

  WRAPPED_PUPable_abstract(Event);  // NOLINT

  template <typename ComputeTagsList, typename DataBoxType,
            typename Metavariables, typename ArrayIndex,
            typename ComponentPointer>
  void run(const ObservationBox<ComputeTagsList, DataBoxType>& box,
           Parallel::GlobalCache<Metavariables>& cache,
           const ArrayIndex& array_index,
           const ComponentPointer /*meta*/) const {
    using factory_classes =
        typename std::decay_t<Metavariables>::factory_creation::factory_classes;
    call_with_dynamic_type<void, tmpl::at<factory_classes, Event>>(
        this, [&box, &cache, &array_index](auto* const event) {
          apply(*event, box, cache, array_index, ComponentPointer{});
        });
  }

  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename ComponentPointer>
  bool is_ready(const db::DataBox<DbTags>& box,
                Parallel::GlobalCache<Metavariables>& cache,
                const ArrayIndex& array_index,
                const ComponentPointer /*meta*/) const {
    using factory_classes =
        typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            box))>::factory_creation::factory_classes;
    return call_with_dynamic_type<bool, tmpl::at<factory_classes, Event>>(
        this, [&box, &cache, &array_index](auto* const event) {
          return db::apply<
              typename std::decay_t<decltype(*event)>::is_ready_argument_tags>(
              [&event, &cache, &array_index](const auto&... args) {
                return event->is_ready(args..., cache, array_index,
                                       ComponentPointer{});
              },
              box);
        });
  }

  /// Whether the event uses anything depending on the
  /// evolved_variables.  If this returns false, anything depending on
  /// the evolved variables may have an incorrect value when the event
  /// is run.
  virtual bool needs_evolved_variables() const = 0;
};
