// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <pup.h>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Utilities/FakeVirtual.hpp"

/// \ingroup EventsAndDenseTriggersGroup
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
 public:
  /// %Result type for the `is_triggered` method.
  ///
  /// This indicates whether the trigger has fired and the next time
  /// the trigger should be checked.  The consumer is not required to
  /// wait until the requested time before testing the trigger again.
  struct Result {
    bool is_triggered;
    double next_check;
  };

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
  explicit DenseTrigger(CkMigrateMessage* const msg) noexcept
      : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(DenseTrigger);  // NOLINT
  /// \endcond

  /// Check whether the trigger fires.
  template <typename DbTags>
  Result is_triggered(const db::DataBox<DbTags>& box) const noexcept {
    using factory_classes =
        typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            box))>::factory_creation::factory_classes;
    return call_with_dynamic_type<Result,
                                  tmpl::at<factory_classes, DenseTrigger>>(
        this, [&box](auto* const trigger) noexcept {
          using TriggerType = std::decay_t<decltype(*trigger)>;
          return db::apply<typename TriggerType::is_triggered_argument_tags>(
              [&trigger](const auto&... args) noexcept {
                return trigger->is_triggered(args...);
              },
              box);
        });
  }

  /// Check whether all data required to evaluate the trigger is
  /// available.  The trigger is not responsible for checking whether
  /// dense output of the evolved variables is possible, but may need
  /// to check things such as the availability of FunctionOfTime data.
  template <typename DbTags>
  bool is_ready(const db::DataBox<DbTags>& box) const noexcept {
    using factory_classes =
        typename std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            box))>::factory_creation::factory_classes;
    return call_with_dynamic_type<bool,
                                  tmpl::at<factory_classes, DenseTrigger>>(
        this, [&box](auto* const trigger) noexcept {
          using TriggerType = std::decay_t<decltype(*trigger)>;
          return db::apply<typename TriggerType::is_ready_argument_tags>(
              [&trigger](const auto&... args) noexcept {
                return trigger->is_ready(args...);
              },
              box);
        });
  }
};
