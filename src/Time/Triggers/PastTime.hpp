// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "Evolution/EventsAndTriggers/Trigger.hpp"
#include "Options/Options.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Triggers {
/// \ingroup EventsAndTriggersGroup
/// \ingroup TimeGroup
/// Trigger when the simulation is past a certain time (after that
/// time if time is running forward, before that time if time is
/// running backward).
template <typename KnownTriggers>
class PastTime : public Trigger<KnownTriggers> {
 public:
  /// \cond
  PastTime() = default;
  explicit PastTime(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(PastTime);  // NOLINT
  /// \endcond

  static constexpr OptionString help{
      "Trigger if the simulation is past a given time."};

  explicit PastTime(const double trigger_time) noexcept
      : trigger_time_(trigger_time) {}

  using argument_tags = tmpl::list<Tags::TimeValue, Tags::TimeStep>;

  bool operator()(const double time, const TimeDelta& time_step) const
      noexcept {
    if (time_step.is_positive()) {
      return time > trigger_time_;
    } else {
      return time < trigger_time_;
    }
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | trigger_time_;
  }

 private:
  double trigger_time_{std::numeric_limits<double>::signaling_NaN()};
};

/// \cond
template <typename KnownTriggers>
PUP::able::PUP_ID PastTime<KnownTriggers>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Triggers

template <typename KnownTriggers>
struct create_from_yaml<Triggers::PastTime<KnownTriggers>> {
  static Triggers::PastTime<KnownTriggers> create(const Option& options) {
    return Triggers::PastTime<KnownTriggers>(options.parse_as<double>());
  }
};
