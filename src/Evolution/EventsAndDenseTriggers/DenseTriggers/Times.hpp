// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
struct TimeStepId;
}  // namespace Tags
/// \endcond

namespace DenseTriggers {
/// \cond
template <typename TriggerRegistrars>
class Times;
/// \endcond

namespace Registrars {
using Times = Registration::Registrar<DenseTriggers::Times>;
}  // namespace Registrars

/// \ingroup EventsAndTriggersGroup
/// \ingroup TimeGroup
/// Trigger at specified times.
template <typename TriggerRegistrars = tmpl::list<Registrars::Times>>
class Times : public DenseTrigger<TriggerRegistrars> {
 public:
  /// \cond
  Times() = default;
  explicit Times(CkMigrateMessage* const msg) noexcept
      : DenseTrigger<TriggerRegistrars>(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Times);  // NOLINT
  /// \endcond

  using Result = typename DenseTrigger<TriggerRegistrars>::Result;

  static constexpr Options::String help{"Trigger at specified times."};

  explicit Times(std::unique_ptr<TimeSequence<double>> times) noexcept
      : times_(std::move(times)) {}

  using is_triggered_argument_tags = tmpl::list<Tags::TimeStepId, Tags::Time>;

  Result is_triggered(const TimeStepId& time_step_id,
                      const double time) const noexcept {
    const evolution_less<double> before{time_step_id.time_runs_forward()};

    const auto trigger_times = times_->times_near(time);
    double next_time = time_step_id.time_runs_forward()
                           ? std::numeric_limits<double>::infinity()
                           : -std::numeric_limits<double>::infinity();
    for (const auto& trigger_time : trigger_times) {
      if (trigger_time.has_value() and before(time, *trigger_time) and
          before(*trigger_time, next_time)) {
        next_time = *trigger_time;
      }
    }

    return {time == trigger_times[1], next_time};
  }

  using is_ready_argument_tags = tmpl::list<>;

  bool is_ready() const noexcept {
    return true;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override {
    DenseTrigger<TriggerRegistrars>::pup(p);
    p | times_;
  }

 private:
  std::unique_ptr<TimeSequence<double>> times_{};
};

/// \cond
template <typename TriggerRegistrars>
PUP::able::PUP_ID Times<TriggerRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace DenseTriggers

template <typename TriggerRegistrars>
struct Options::create_from_yaml<DenseTriggers::Times<TriggerRegistrars>> {
  template <typename Metavariables>
  static DenseTriggers::Times<TriggerRegistrars> create(const Option& options) {
    return DenseTriggers::Times<TriggerRegistrars>(
        options.parse_as<std::unique_ptr<TimeSequence<double>>>());
  }
};
