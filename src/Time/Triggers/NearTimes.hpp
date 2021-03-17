// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <utility>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeSequence.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
struct TimeStep;
}  // namespace Tags
/// \endcond

namespace Triggers {
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
/// \see Times
class NearTimes : public Trigger {
 public:
  /// \cond
  NearTimes() = default;
  explicit NearTimes(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(NearTimes);  // NOLINT
  /// \endcond

  using Unit = NearTimes_enums::Unit;
  using Direction = NearTimes_enums::Direction;

  struct OptionTags {
    struct Times {
      using type = std::unique_ptr<TimeSequence<double>>;
      static constexpr Options::String help = "Times to trigger at";
    };

    struct Range {
      using type = double;
      static type lower_bound() noexcept { return 0.0; }
      static constexpr Options::String help =
          "Maximum time difference to trigger at";
    };

    struct Unit {
      using type = NearTimes::Unit;
      static constexpr Options::String help =
          "Interpret Range as 'Time', 'Step's, or 'Slab's";
    };

    struct Direction {
      using type = NearTimes::Direction;
      static constexpr Options::String help =
          "Trigger 'Before', 'After', or 'Both' from the times";
    };
  };

  static constexpr Options::String help =
      "Trigger in intervals surrounding particular times.";
  using options =
      tmpl::list<typename OptionTags::Times, typename OptionTags::Range,
                 typename OptionTags::Unit, typename OptionTags::Direction>;

  NearTimes(std::unique_ptr<TimeSequence<double>> times, const double range,
            const Unit unit, const Direction direction) noexcept
      : times_(std::move(times)),
        range_(range),
        unit_(unit),
        direction_(direction) {}

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

    const auto nearby_times = times_->times_near(trigger_range.first);
    for (const auto& time : nearby_times) {
      if (time and *time >= trigger_range.first and
          *time <= trigger_range.second) {
        return true;
      }
    }
    return false;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override {
    p | times_;
    p | range_;
    p | unit_;
    p | direction_;
  }

 private:
  std::unique_ptr<TimeSequence<double>> times_{};
  double range_ = std::numeric_limits<double>::signaling_NaN();
  Unit unit_{};
  Direction direction_{};
};
}  // namespace Triggers

template <>
struct Options::create_from_yaml<Triggers::NearTimes_enums::Unit> {
  using type = Triggers::NearTimes_enums::Unit;
  template <typename Metavariables>
  static type create(const Options::Option& options) {
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
struct Options::create_from_yaml<
    typename Triggers::NearTimes_enums::Direction> {
  using type = Triggers::NearTimes_enums::Direction;
  template <typename Metavariables>
  static type create(const Options::Option& options) {
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
