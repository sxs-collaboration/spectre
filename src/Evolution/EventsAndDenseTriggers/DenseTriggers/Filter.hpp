// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

namespace DenseTriggers {
/// \cond
template <typename TriggerRegistrars, typename DenseTriggerRegistrars>
class Filter;
/// \endcond

namespace Registrars {
template <typename TriggerRegistrars>
using Filter =
    Registration::Registrar<DenseTriggers::Filter, TriggerRegistrars>;
}  // namespace Registrars

/// \ingroup EventsAndTriggersGroup
/// %Filter activations of a dense trigger using a non-dense trigger.
///
/// For example, to trigger every 10 starting at 100, one could use
///
/// \snippet Test_Filter.cpp example
template <typename TriggerRegistrars, typename DenseTriggerRegistrars>
class Filter : public DenseTrigger<DenseTriggerRegistrars> {
 public:
  /// \cond
  Filter() = default;
  explicit Filter(CkMigrateMessage* const msg) noexcept
      : DenseTrigger<DenseTriggerRegistrars>(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Filter);  // NOLINT
  /// \endcond

  using Result = typename DenseTrigger<DenseTriggerRegistrars>::Result;

  struct TriggerOption {
    static std::string name() noexcept { return "Trigger"; }
    using type = std::unique_ptr<DenseTrigger<DenseTriggerRegistrars>>;
    static constexpr Options::String help = "Dense trigger to filter";
  };

  struct FilterOption {
    static std::string name() noexcept { return "Filter"; }
    using type = std::unique_ptr<Trigger<TriggerRegistrars>>;
    static constexpr Options::String help = "Non-dense trigger to filter with";
  };

  using options = tmpl::list<TriggerOption, FilterOption>;
  static constexpr Options::String help =
      "Filter activations of a dense trigger using a non-dense trigger.";

  explicit Filter(std::unique_ptr<DenseTrigger<DenseTriggerRegistrars>> trigger,
                  std::unique_ptr<Trigger<TriggerRegistrars>> filter) noexcept
      : trigger_(std::move(trigger)), filter_(std::move(filter)) {}

  using is_triggered_argument_tags = tmpl::list<Tags::DataBox>;

  template <typename DbTags>
  Result is_triggered(const db::DataBox<DbTags>& box) const noexcept {
    auto result = trigger_->is_triggered(box);
    if (not filter_->is_triggered(box)) {
      result.is_triggered = false;
    }
    return result;
  }

  using is_ready_argument_tags = tmpl::list<Tags::DataBox>;

  template <typename DbTags>
  bool is_ready(const db::DataBox<DbTags>& box) const noexcept {
    return trigger_->is_ready(box);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override {
    DenseTrigger<DenseTriggerRegistrars>::pup(p);
    p | trigger_;
    p | filter_;
  }

 private:
  std::unique_ptr<DenseTrigger<DenseTriggerRegistrars>> trigger_{};
  std::unique_ptr<Trigger<TriggerRegistrars>> filter_{};
};

/// \cond
template <typename TriggerRegistrars, typename DenseTriggerRegistrars>
PUP::able::PUP_ID Filter<TriggerRegistrars, DenseTriggerRegistrars>::my_PUP_ID =
    0;  // NOLINT
/// \endcond
}  // namespace DenseTriggers
