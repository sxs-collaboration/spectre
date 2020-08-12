// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <pup.h>
#include <utility>
#include <vector>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/Utilities.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
struct TimeStepId;
}  // namespace Tags
/// \endcond

namespace Triggers {
template <typename TriggerRegistrars>
class SpecifiedTimes;

namespace Registrars {
using SpecifiedTimes = Registration::Registrar<Triggers::SpecifiedTimes>;
}  // namespace Registrars

/// \ingroup EventsAndTriggersGroup
/// \ingroup TimeGroup
/// Trigger at particular times.
///
/// \warning This trigger will only fire if it is actually checked at
/// the times specified.  The StepToTimes StepChooser can be useful
/// for this.
template <typename TriggerRegistrars = tmpl::list<Registrars::SpecifiedTimes>>
class SpecifiedTimes : public Trigger<TriggerRegistrars> {
 public:
  /// \cond
  SpecifiedTimes() = default;
  explicit SpecifiedTimes(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SpecifiedTimes);  // NOLINT
  /// \endcond

  struct Times {
    using type = std::vector<double>;
    static constexpr OptionString help{"Times to trigger at"};
  };

  static constexpr OptionString help{"Trigger at particular times."};
  using options = tmpl::list<Times>;

  explicit SpecifiedTimes(std::vector<double> times) noexcept
      : times_(std::move(times)) {
    std::sort(times_.begin(), times_.end());
  }

  using argument_tags = tmpl::list<Tags::Time, Tags::TimeStepId>;

  bool operator()(const double now, const TimeStepId& time_id) const noexcept {
    const auto& substep_time = time_id.substep_time();
    // Trying to step to a given time might not get us exactly there
    // because of rounding errors.
    const double sloppiness = slab_rounding_error(substep_time);

    const auto triggered_times = std::equal_range(
        times_.begin(), times_.end(), now,
        [&sloppiness](const double a, const double b) noexcept {
          return a < b - sloppiness;
        });
    return triggered_times.first != triggered_times.second;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override { p | times_; }

 private:
  std::vector<double> times_;
};

/// \cond
template <typename TriggerRegistrars>
PUP::able::PUP_ID SpecifiedTimes<TriggerRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Triggers
