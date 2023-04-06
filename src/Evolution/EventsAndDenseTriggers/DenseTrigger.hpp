// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <optional>
#include <pup.h>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Time/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

/// \ingroup EventsAndTriggersGroup
namespace DenseTriggers {}

/// \ingroup EventsAndTriggersGroup
/// Base class for checking whether to run an Event at arbitrary times.
///
/// The DataBox passed to the member functions will have
/// `::Tags::Time`, and therefore any compute tags depending on that
/// value, set to the time to be tested.  Any discrete properties of
/// steps or slabs, such as the step size, may have the values from
/// times off by one step.  The evolved variables will be in an
/// unspecified state.
class DenseTrigger : public PUP::able {
 protected:
  /// \cond
  DenseTrigger() = default;
  DenseTrigger(const DenseTrigger&) = default;
  DenseTrigger(DenseTrigger&&) = default;
  DenseTrigger& operator=(const DenseTrigger&) = default;
  DenseTrigger& operator=(DenseTrigger&&) = default;
  /// \endcond

 public:
  ~DenseTrigger() override = default;

  /// \cond
  explicit DenseTrigger(CkMigrateMessage* const msg) : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(DenseTrigger);  // NOLINT
  /// \endcond

  /// Check whether the trigger fires.  Returns std::nullopt if
  /// insufficient data is available to make the decision.  The
  /// trigger is not responsible for checking whether dense output of
  /// the evolved variables is possible, but may need to check things
  /// such as the availability of FunctionOfTime data.
  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename Component>
  std::optional<bool> is_triggered(const db::DataBox<DbTags>& box,
                                   Parallel::GlobalCache<Metavariables>& cache,
                                   const ArrayIndex& array_index,
                                   const Component* const component) {
    using factory_classes =
        typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            box))>::factory_creation::factory_classes;
    previous_trigger_time_ = next_previous_trigger_time_;
    return call_with_dynamic_type<std::optional<bool>,
                                  tmpl::at<factory_classes, DenseTrigger>>(
        this,
        [this, &array_index, &box, &cache, &component](auto* const trigger) {
          using TriggerType = std::decay_t<decltype(*trigger)>;
          const auto result =
              db::apply<typename TriggerType::is_triggered_argument_tags>(
                  [&array_index, &cache, &component,
                   &trigger](const auto&... args) {
                    return trigger->is_triggered(cache, array_index, component,
                                                 args...);
                  },
                  box);
          if (result == std::optional{true}) {
            next_previous_trigger_time_ = db::get<::Tags::Time>(box);
          }
          return result;
        });
  }

  /// Obtain the next time to check the trigger, or std::nullopt if the
  /// trigger is not ready to report yet.
  template <typename DbTags, typename Metavariables, typename ArrayIndex,
            typename Component>
  std::optional<double> next_check_time(
      const db::DataBox<DbTags>& box,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const Component* component) {
    using factory_classes =
        typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            box))>::factory_creation::factory_classes;
    return call_with_dynamic_type<std::optional<double>,
                                  tmpl::at<factory_classes, DenseTrigger>>(
        this, [&array_index, &box, &cache, &component](auto* const trigger) {
          using TriggerType = std::decay_t<decltype(*trigger)>;
          return db::apply<typename TriggerType::next_check_time_argument_tags>(
              [&array_index, &cache, &component,
               &trigger](const auto&... args) {
                return trigger->next_check_time(cache, array_index, component,
                                                args...);
              },
              box);
        });
  }

  /// \brief Reports the value of `::Tags::Time` when the trigger most recently
  /// fired, except for the most recent call of `is_triggered`.
  ///
  /// \details The most recent call of `is_triggered` is not used for reporting
  /// the previous trigger so that the time reported to the event is actually
  /// the previous time value on which the trigger fired and activated the
  /// event. Without ignoring the most recent call of `is_triggered`, we'd just
  /// always be reporting the current time to the event, because events always
  /// run after their associated triggers fire via a call to `is_triggered`.
  std::optional<double> previous_trigger_time() const {
    return previous_trigger_time_;
  }

  void pup(PUP::er& p) override {
    p | next_previous_trigger_time_;
    p | previous_trigger_time_;
  }

 private:
  std::optional<double> next_previous_trigger_time_ = std::nullopt;
  std::optional<double> previous_trigger_time_ = std::nullopt;
};
