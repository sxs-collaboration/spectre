// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class TimeStepId.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>

#include "ErrorHandling/Assert.hpp"
#include "Time/Time.hpp"

namespace PUP {
class er;
}  // namespace PUP

/// \ingroup TimeGroup
///
/// A unique identifier for the temporal state of an integrated
/// system.
class TimeStepId {
 public:
  TimeStepId() = default;
  /// Create a TimeStepId at the start of a step.  If that step is at the
  /// (evolution-defined) end of the slab the TimeStepId will be advanced
  /// to the next slab.
  TimeStepId(const bool time_runs_forward, const int64_t slab_number,
             const Time& time) noexcept
      : time_runs_forward_(time_runs_forward),
        slab_number_(slab_number),
        step_time_(time),
        substep_time_(time) {
    canonicalize();
  }
  /// Create a TimeStepId at a substep at time `substep_time` in a step
  /// starting at time `step_time`.
  TimeStepId(const bool time_runs_forward, const int64_t slab_number,
             const Time& step_time, const uint64_t substep,
             const Time& substep_time) noexcept
      : time_runs_forward_(time_runs_forward),
        slab_number_(slab_number),
        step_time_(step_time),
        substep_(substep),
        substep_time_(substep_time) {
    ASSERT(substep_ != 0 or step_time_ == substep_time_,
           "Initial substep must align with the step.");
    canonicalize();
  }

  bool time_runs_forward() const noexcept { return time_runs_forward_; }
  int64_t slab_number() const noexcept { return slab_number_; }
  /// Time at the start of the current step
  const Time& step_time() const noexcept { return step_time_; }
  uint64_t substep() const noexcept { return substep_; }
  /// Time of the current substep
  const Time& substep_time() const noexcept { return substep_time_; }

  bool is_at_slab_boundary() const noexcept {
    return substep_ == 0 and substep_time_.is_at_slab_boundary();
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  void canonicalize() noexcept;

  bool time_runs_forward_{false};
  int64_t slab_number_{0};
  Time step_time_{};
  uint64_t substep_{0};
  Time substep_time_{};
};

bool operator==(const TimeStepId& a, const TimeStepId& b) noexcept;
bool operator!=(const TimeStepId& a, const TimeStepId& b) noexcept;
bool operator<(const TimeStepId& a, const TimeStepId& b) noexcept;
bool operator<=(const TimeStepId& a, const TimeStepId& b) noexcept;
bool operator>(const TimeStepId& a, const TimeStepId& b) noexcept;
bool operator>=(const TimeStepId& a, const TimeStepId& b) noexcept;

std::ostream& operator<<(std::ostream& s, const TimeStepId& id) noexcept;

size_t hash_value(const TimeStepId& id) noexcept;

namespace std {
template <>
struct hash<TimeStepId> {
  size_t operator()(const TimeStepId& id) const noexcept;
};
}  // namespace std
