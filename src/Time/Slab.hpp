// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Slab

#pragma once

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <limits>

#include "ErrorHandling/Assert.hpp"

class Time;
class TimeDelta;

namespace PUP {
class er;
}  // namespace PUP

/// \ingroup TimeGroup
///
/// A chunk of time.  Every element must reach slab boundaries
/// exactly, no matter how it actually takes time steps to get there.
/// The simulation can only be assumed to have global data available
/// at slab boundaries.
class Slab {
 public:
  /// Default constructor gives an invalid Slab.
  Slab() noexcept
      : start_(std::numeric_limits<double>::signaling_NaN()),
        end_(std::numeric_limits<double>::signaling_NaN()) {}

  /// Construct a slab running between two times (exactly).
  Slab(double start, double end) noexcept : start_(start), end_(end) {
    ASSERT(start_ < end_, "Backwards Slab");
  }

  /// Construct a slab with a given start time and duration.  The
  /// actual duration may differ by roundoff from the supplied value.
  static Slab with_duration_from_start(double start, double duration) noexcept {
    return {start, start + duration};
  }

  /// Construct a slab with a given end time and duration.  The
  /// actual duration may differ by roundoff from the supplied value.
  static Slab with_duration_to_end(double end, double duration) noexcept {
    return {end - duration, end};
  }

  Time start() const noexcept;
  Time end() const noexcept;
  TimeDelta duration() const noexcept;

  /// Create a new slab immediately following this one with the same
  /// (up to roundoff) duration.
  Slab advance() const noexcept { return {end_, end_ + (end_ - start_)}; }

  /// Create a new slab immediately preceeding this one with the same
  /// (up to roundoff) duration.
  Slab retreat() const noexcept { return {start_ - (end_ - start_), start_}; }

  /// Create a slab adjacent to this one in the direction indicated by
  /// the argument, as with advance() or retreat().
  Slab advance_towards(const TimeDelta& dt) const noexcept;

  /// Create a new slab with the same start time as this one with the
  /// given duration (up to roundoff).
  Slab with_duration_from_start(double duration) const noexcept {
    return {start_, start_ + duration};
  }

  /// Create a new slab with the same end time as this one with the
  /// given duration (up to roundoff).
  Slab with_duration_to_end(double duration) const noexcept {
    return {end_ - duration, end_};
  }

  /// Check if this slab is immediately followed by the other slab.
  bool is_followed_by(const Slab& other) const noexcept {
    return end_ == other.start_;
  }

  /// Check if this slab is immediately preceeded by the other slab.
  bool is_preceeded_by(const Slab& other) const noexcept {
    return other.is_followed_by(*this);
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  double start_;
  double end_;

  friend class Time;
  friend class TimeDelta;

  friend bool operator==(const Slab& a, const Slab& b) noexcept;
  friend bool operator<(const Slab& a, const Slab& b) noexcept;
  friend bool operator==(const Time& a, const Time& b) noexcept;
};

inline bool operator==(const Slab& a, const Slab& b) noexcept {
  return a.start_ == b.start_ and a.end_ == b.end_;
}
inline bool operator!=(const Slab& a, const Slab& b) noexcept {
  return not(a == b);
}

/// Slab comparison operators give the time ordering.  Overlapping
/// unequal slabs should not be compared (and will trigger an
/// assertion).
//@{
inline bool operator<(const Slab& a, const Slab& b) noexcept {
  ASSERT(a == b or a.end_ <= b.start_ or a.start_ >= b.end_,
         "Cannot compare overlapping slabs");
  return a.end_ <= b.start_;
}
inline bool operator>(const Slab& a, const Slab& b) noexcept {
  return b < a;
}
inline bool operator<=(const Slab& a, const Slab& b) noexcept {
  return not(a > b);
}
inline bool operator>=(const Slab& a, const Slab& b) noexcept {
  return not(a < b);
}
//@}

std::ostream& operator<<(std::ostream& os, const Slab& s) noexcept;

size_t hash_value(const Slab& s) noexcept;

namespace std {
template <>
struct hash<Slab> {
  size_t operator()(const Slab& s) const noexcept;
};
}  // namespace std
