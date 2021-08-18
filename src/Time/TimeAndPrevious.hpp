// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <functional>
#include <optional>
#include <ostream>

#include "Parallel/PupStlCpp17.hpp"
#include "Utilities/StdHelpers.hpp"

/*!
 * \brief Used for storing both the current time and a previous time
 *
 * \details The intended use case is as an alternative temporal id for use as a
 * key in `std::map`s or `std::unordered_map`s. This is primarily useful for
 * events used in `EventsAndDenseTriggers`, which has the ability to provide the
 * previous trigger time for verifying sequencing of communication.
 */
struct TimeAndPrevious {
  double time;
  std::optional<double> previous_time;

  void pup(PUP::er& p) noexcept {
    p | time;
    p | previous_time;
  }
};

inline bool operator==(const TimeAndPrevious& lhs, const TimeAndPrevious& rhs) {
  return lhs.time == rhs.time and lhs.previous_time == rhs.previous_time;
}

inline bool operator!=(const TimeAndPrevious& lhs, const TimeAndPrevious& rhs) {
  return not(lhs == rhs);
}

inline std::ostream& operator<<(std::ostream& s,
                                const TimeAndPrevious& time_and_previous) {
  return s << time_and_previous.time << ", " << time_and_previous.previous_time;
}

struct TimeAndPreviousLessComparator {
  using is_transparent = std::true_type;
  bool operator()(const double time,
                  const TimeAndPrevious& time_and_previous) const noexcept {
    return time < time_and_previous.time;
  }
  bool operator()(const TimeAndPrevious& lhs,
                  const TimeAndPrevious& rhs) const noexcept {
    return lhs.time < rhs.time or
           (lhs.time == rhs.time and lhs.previous_time < rhs.previous_time);
  }
  bool operator()(const TimeAndPrevious& time_and_previous,
                  const double time) const noexcept {
    return time_and_previous.time < time;
  }
};

// hash definition for use of TimeAndPrevious in STL hash tables
// `std::unordered_map` and `std::unordered_set`.
namespace std {
template <>
struct hash<TimeAndPrevious> {
  size_t operator()(const TimeAndPrevious& time_and_previous) const noexcept {
    return hash_(time_and_previous.time);
  }

 private:
  std::hash<double> hash_;
};
}  // namespace std
