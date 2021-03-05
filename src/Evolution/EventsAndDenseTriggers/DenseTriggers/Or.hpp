// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct TimeStepId;
}  // namespace Tags
/// \endcond

namespace DenseTriggers {
/// \cond
template <typename TriggerRegistrars>
class Or;
/// \endcond

namespace Registrars {
using Or = Registration::Registrar<DenseTriggers::Or>;
}  // namespace Registrars

/// \ingroup EventsAndTriggersGroup
/// Trigger when any of a collection of DenseTriggers triggers.
template <typename TriggerRegistrars>
class Or : public DenseTrigger<TriggerRegistrars> {
 public:
  /// \cond
  Or() = default;
  explicit Or(CkMigrateMessage* const msg) noexcept
      : DenseTrigger<TriggerRegistrars>(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Or);  // NOLINT
  /// \endcond

  using Result = typename DenseTrigger<TriggerRegistrars>::Result;

  static constexpr Options::String help =
      "Trigger when any of a collection of triggers triggers.";

  explicit Or(std::vector<std::unique_ptr<DenseTrigger<TriggerRegistrars>>>
                  triggers) noexcept
      : triggers_(std::move(triggers)) {}

  using is_triggered_argument_tags =
      tmpl::list<Tags::TimeStepId, Tags::DataBox>;

  template <typename DbTags>
  Result is_triggered(const TimeStepId& time_step_id,
                      const db::DataBox<DbTags>& box) const noexcept {
    const evolution_less<double> before{time_step_id.time_runs_forward()};
    Result result{false, time_step_id.time_runs_forward()
                             ? std::numeric_limits<double>::infinity()
                             : -std::numeric_limits<double>::infinity()};
    for (const auto& trigger : triggers_) {
      const auto sub_result = trigger->is_triggered(box);
      if (sub_result.is_triggered) {
        // We can't short-circuit because we need to make sure we
        // report the next time that any of the triggers wants to be
        // checked, whether they triggered now or not.
        result.is_triggered = true;
      }
      result.next_check =
          std::min(sub_result.next_check, result.next_check, before);
    }
    return result;
  }

  using is_ready_argument_tags = tmpl::list<Tags::DataBox>;

  template <typename DbTags>
  bool is_ready(const db::DataBox<DbTags>& box) const noexcept {
    return alg::all_of(
        triggers_,
        [&box](const std::unique_ptr<DenseTrigger<TriggerRegistrars>>&
                   trigger) noexcept { return trigger->is_ready(box); });
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override {
    DenseTrigger<TriggerRegistrars>::pup(p);
    p | triggers_;
  }

 private:
  std::vector<std::unique_ptr<DenseTrigger<TriggerRegistrars>>> triggers_{};
};

/// \cond
template <typename TriggerRegistrars>
PUP::able::PUP_ID Or<TriggerRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace DenseTriggers

template <typename TriggerRegistrars>
struct Options::create_from_yaml<DenseTriggers::Or<TriggerRegistrars>> {
  template <typename Metavariables>
  static DenseTriggers::Or<TriggerRegistrars> create(const Option& options) {
    return DenseTriggers::Or<TriggerRegistrars>(
        options.parse_as<
            std::vector<std::unique_ptr<DenseTrigger<TriggerRegistrars>>>>());
  }
};
