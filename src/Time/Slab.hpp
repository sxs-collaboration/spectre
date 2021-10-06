// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Slab

#pragma once

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <limits>

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
  Slab() = default;

  /// Construct a slab running between two times (exactly).
  Slab(double start, double end);

  /// Construct a slab with a given start time and duration.  The
  /// actual duration may differ by roundoff from the supplied value.
  static Slab with_duration_from_start(double start, double duration);

  /// Construct a slab with a given end time and duration.  The
  /// actual duration may differ by roundoff from the supplied value.
  static Slab with_duration_to_end(double end, double duration);

  Time start() const;
  Time end() const;
  TimeDelta duration() const;

  /// Create a new slab immediately following this one with the same
  /// (up to roundoff) duration.
  Slab advance() const;

  /// Create a new slab immediately preceeding this one with the same
  /// (up to roundoff) duration.
  Slab retreat() const;

  /// Create a slab adjacent to this one in the direction indicated by
  /// the argument, as with advance() or retreat().
  Slab advance_towards(const TimeDelta& dt) const;

  /// Create a new slab with the same start time as this one with the
  /// given duration (up to roundoff).
  Slab with_duration_from_start(double duration) const;

  /// Create a new slab with the same end time as this one with the
  /// given duration (up to roundoff).
  Slab with_duration_to_end(double duration) const;

  /// Check if this slab is immediately followed by the other slab.
  bool is_followed_by(const Slab& other) const;

  /// Check if this slab is immediately preceeded by the other slab.
  bool is_preceeded_by(const Slab& other) const;

  /// Check if slabs overlap.  Abutting slabs do not overlap.
  bool overlaps(const Slab& other) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  double start_ = std::numeric_limits<double>::signaling_NaN();
  double end_ = std::numeric_limits<double>::signaling_NaN();

  friend class Time;
  friend class TimeDelta;

  friend bool operator==(const Slab& a, const Slab& b);
  friend bool operator<(const Slab& a, const Slab& b);
  friend bool operator==(const Time& a, const Time& b);
};

bool operator!=(const Slab& a, const Slab& b);

/// Slab comparison operators give the time ordering.  Overlapping
/// unequal slabs should not be compared (and will trigger an
/// assertion).
/// @{
// NOLINTNEXTLINE(readability-redundant-declaration) redeclared for docs
bool operator<(const Slab& a, const Slab& b);
bool operator>(const Slab& a, const Slab& b);
bool operator<=(const Slab& a, const Slab& b);
bool operator>=(const Slab& a, const Slab& b);
/// @}

std::ostream& operator<<(std::ostream& os, const Slab& s);

size_t hash_value(const Slab& s);

namespace std {
template <>
struct hash<Slab> {
  size_t operator()(const Slab& s) const;
};
}  // namespace std
