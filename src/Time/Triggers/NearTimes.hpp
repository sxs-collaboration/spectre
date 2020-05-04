// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <pup.h>
#include <string>
#include <utility>
#include <vector>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
struct TimeStep;
}  // namespace Tags
/// \endcond

namespace Triggers {
template <typename TriggerRegistrars>
class NearTimes;

namespace Registrars {
using NearTimes = Registration::Registrar<Triggers::NearTimes>;
}  // namespace Registrars

namespace NearTimes_enums {
enum class Unit { Time, Slab, Step };
enum class Direction { Before, After, Both };
}  // namespace NearTimes_enums

/// \ingroup EventsAndTriggersGroup
/// \ingroup TimeGroup
/// Trigger in intervals surrounding particular times.
///
/// When using adaptive time stepping, intervals specified in terms of
/// slabs or steps are approximate.
///
/// \see SpecifiedTimes
template <typename TriggerRegistrars = tmpl::list<Registrars::NearTimes>>
class NearTimes : public Trigger<TriggerRegistrars> {
 public:
  /// \cond
  NearTimes() = default;
  explicit NearTimes(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(NearTimes);  // NOLINT
  /// \endcond

  using Unit = NearTimes_enums::Unit;
  using Direction = NearTimes_enums::Direction;

  struct Options {
    struct Times {
      using type = std::vector<double>;
      static constexpr OptionString help = "Times to trigger at";
    };

    struct Range {
      using type = double;
      static type lower_bound() noexcept { return 0.0; }
      static constexpr OptionString help =
          "Maximum time difference to trigger at";
    };

    struct Unit {
      using type = NearTimes::Unit;
      static constexpr OptionString help =
          "Interpret Range as 'Time', 'Step's, or 'Slab's";
    };

    struct Direction {
      using type = NearTimes::Direction;
      static constexpr OptionString help =
          "Trigger 'Before', 'After', or 'Both' from the times";
    };
  };

  static constexpr OptionString help =
      "Trigger in intervals surrounding particular times.";
  using options =
      tmpl::list<typename Options::Times, typename Options::Range,
                 typename Options::Unit, typename Options::Direction>;

  NearTimes(std::vector<double> times, const double range, const Unit unit,
            const Direction direction) noexcept
      : times_(std::move(times)),
        range_(range),
        unit_(unit),
        direction_(direction) {
    std::sort(times_.begin(), times_.end());
  }

  using argument_tags = tmpl::list<Tags::Time, Tags::TimeStep>;

  bool operator()(const double now, const TimeDelta& time_step) const noexcept {
    const bool time_runs_forward = time_step.is_positive();

    double range_code_units = range_;
    if (unit_ == Unit::Slab) {
      range_code_units *= time_step.slab().duration().value();
    } else if (unit_ == Unit::Step) {
      range_code_units *= std::abs(time_step.value());
    }

    if (not time_runs_forward) {
      range_code_units = -range_code_units;
    }

    // Interval around now to look for trigger times in.
    auto trigger_range = std::make_pair(
        direction_ == Direction::Before ? now : now - range_code_units,
        direction_ == Direction::After ? now : now + range_code_units);

    if (not time_runs_forward) {
      std::swap(trigger_range.first, trigger_range.second);
    }

    const auto next_time =
        std::lower_bound(times_.begin(), times_.end(), trigger_range.first);
    return next_time != times_.end() and *next_time <= trigger_range.second;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override {
    p | times_;
    p | range_;
    p | unit_;
    p | direction_;
  }

 private:
  std::vector<double> times_{};
  double range_ = std::numeric_limits<double>::signaling_NaN();
  Unit unit_{};
  Direction direction_{};
};

/// \cond
template <typename TriggerRegistrars>
PUP::able::PUP_ID NearTimes<TriggerRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Triggers

template <>
struct create_from_yaml<Triggers::NearTimes_enums::Unit> {
  using type = Triggers::NearTimes_enums::Unit;
  template <typename Metavariables>
  static type create(const Option& options) {
    const auto unit = options.parse_as<std::string>();
    if (unit == "Time") {
      return type::Time;
    } else if (unit == "Step") {
      return type::Step;
    } else if (unit == "Slab") {
      return type::Slab;
    } else {
      PARSE_ERROR(options.context(), "Unit must be 'Time', 'Step', or 'Slab'");
    }
  }
};

template <>
struct create_from_yaml<typename Triggers::NearTimes_enums::Direction> {
  using type = Triggers::NearTimes_enums::Direction;
  template <typename Metavariables>
  static type create(const Option& options) {
    const auto unit = options.parse_as<std::string>();
    if (unit == "Before") {
      return type::Before;
    } else if (unit == "After") {
      return type::After;
    } else if (unit == "Both") {
      return type::Both;
    } else {
      PARSE_ERROR(options.context(),
                  "Direction must be 'Before', 'After', or 'Both'");
    }
  }
};
