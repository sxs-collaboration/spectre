// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <pup.h>

#include "Evolution/EventsAndTriggers/Trigger.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
struct TimeId;
}  // namespace Tags
/// \endcond

namespace Triggers {
template <typename TriggerRegistrars>
class PastTime;

namespace Registrars {
using PastTime = Registration::Registrar<Triggers::PastTime>;
}  // namespace Registrars

/// \ingroup EventsAndTriggersGroup
/// \ingroup TimeGroup
/// Trigger when the simulation is past a certain time (after that
/// time if time is running forward, before that time if time is
/// running backward).
template <typename TriggerRegistrars = tmpl::list<Registrars::PastTime>>
class PastTime : public Trigger<TriggerRegistrars> {
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

  using argument_tags = tmpl::list<Tags::Time, Tags::TimeId>;

  bool operator()(const Time& time, const TimeId& time_id) const noexcept {
    if (not time_id.is_at_slab_boundary()) {
      return false;
    }
    return evolution_greater<double>{time_id.time_runs_forward()}(
        time.value(), trigger_time_);
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | trigger_time_;
  }

 private:
  double trigger_time_{std::numeric_limits<double>::signaling_NaN()};
};

/// \cond
template <typename TriggerRegistrars>
PUP::able::PUP_ID PastTime<TriggerRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Triggers

template <typename TriggerRegistrars>
struct create_from_yaml<Triggers::PastTime<TriggerRegistrars>> {
  template <typename Metavariables>
  static Triggers::PastTime<TriggerRegistrars> create(const Option& options) {
    return Triggers::PastTime<TriggerRegistrars>(options.parse_as<double>());
  }
};
