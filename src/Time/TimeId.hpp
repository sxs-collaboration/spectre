// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class TimeId.

#pragma once

#include <iosfwd>

#include "Time/Time.hpp"

namespace PUP {
class er;
}  // namespace PUP

/// \ingroup TimeGroup
///
/// A unique identifier for the temporal state of an integrated
/// system.
struct TimeId {
  size_t slab_number{0};
  Time time{};
  size_t substep{0};

  bool is_at_slab_boundary() const noexcept {
    return substep == 0 and time.is_at_slab_boundary();
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT
};

inline bool operator==(const TimeId& a, const TimeId& b) noexcept {
  return a.slab_number == b.slab_number
      and a.time == b.time
      and a.substep == b.substep;
}

inline bool operator!=(const TimeId& a, const TimeId& b) noexcept {
  return not (a == b);
}

std::ostream& operator<<(std::ostream& s, const TimeId& id) noexcept;

size_t hash_value(const TimeId& id) noexcept;

namespace std {
template <>
struct hash<TimeId> {
  size_t operator()(const TimeId& id) const noexcept;
};
}  // namespace std
