// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines Time and TimeDelta

#pragma once

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <limits>
#include <utility>

#include "ErrorHandling/Assert.hpp"
#include "Time/Slab.hpp"
#include "Utilities/Rational.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
class TimeDelta;
/// \endcond

/// \ingroup TimeGroup
///
/// The time in a simulation.  Times can be safely compared for exact
/// equality as long as they do not belong to overlapping unequal
/// slabs.
class Time {
 public:
  using rational_t = Rational;

  /// Default constructor gives an invalid Time.
  Time() noexcept : fraction_(0) {}

  /// A time a given fraction of the way through the given slab.
  Time(Slab slab, rational_t fraction) noexcept
      // clang-tidy: move trivially copyable type
      : slab_(std::move(slab)), fraction_(std::move(fraction)) {  // NOLINT
    range_check();
    compute_value();
  }

  /// Move the time to a different slab.  The time must be at an end
  /// of the current slab and the new slab must share that endpoint.
  Time with_slab(const Slab& new_slab) const noexcept;

  /// Approximate numerical value of the Time.
  double value() const noexcept { return value_; }
  const Slab& slab() const noexcept { return slab_; }
  const rational_t& fraction() const noexcept { return fraction_; }

  Time& operator+=(const TimeDelta& delta) noexcept;
  Time& operator-=(const TimeDelta& delta) noexcept;

  bool is_at_slab_start() const noexcept { return fraction_ == 0; }
  bool is_at_slab_end() const noexcept { return fraction_ == 1; }
  bool is_at_slab_boundary() const noexcept {
    return is_at_slab_start() or is_at_slab_end();
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

  /// A comparison operator that compares Times structurally, i.e.,
  /// just looking at the class members.  This is only intended for
  /// use as the comparator in a map.  The returned ordering does not
  /// match the time ordering and opposite sides of slab boundaries do
  /// not compare equal.  It is, however, much faster to compute than
  /// the temporal ordering, so it is useful when an ordering is
  /// required, but the ordering does not have to be physically
  /// meaningful.
  struct StructuralCompare {
    bool operator()(const Time& a, const Time& b) const {
      if (a.fraction().numerator() != b.fraction().numerator()) {
        return a.fraction().numerator() < b.fraction().numerator();
      }
      if (a.fraction().denominator() != b.fraction().denominator()) {
        return a.fraction().denominator() < b.fraction().denominator();
      }
      return a.slab() < b.slab();
    }
  };

 private:
  Slab slab_;
  rational_t fraction_;
  double value_ = std::numeric_limits<double>::signaling_NaN();

  // The value is precomputed so that we can avoid doing the rational
  // math repeatedly.  The value of a Time should almost always be
  // needed at some point.
  void compute_value() noexcept;

  inline void range_check() const noexcept {
    ASSERT(fraction_ >= 0 and fraction_ <= 1,
           "Out of range slab fraction: " << fraction_);
  }

  friend class TimeDelta;
};

/// \ingroup TimeGroup
///
/// Represents an interval of time within a single slab.
class TimeDelta {
 public:
  using rational_t = Time::rational_t;

  /// Default constructor gives an invalid TimeDelta.
  TimeDelta() noexcept : fraction_(0) {}

  /// An interval covering a given fraction of the slab.
  TimeDelta(Slab slab, rational_t fraction) noexcept
      // clang-tidy: move trivially copyable type
      : slab_(std::move(slab)), fraction_(std::move(fraction)) {}  // NOLINT

  /// Move the interval to a different slab.  The resulting interval
  /// will in general not be the same length, but will take up the
  /// same fraction of its slab.
  TimeDelta with_slab(const Slab& new_slab) const noexcept {
    return {new_slab, fraction()};
  }

  Slab slab() const noexcept { return slab_; }
  rational_t fraction() const noexcept { return fraction_; }

  /// Approximate numerical length of the interval.
  double value() const noexcept {
    return (slab_.end_ - slab_.start_) * fraction_.value();
  }

  /// Test if the interval is oriented towards larger time.
  bool is_positive() const noexcept { return fraction_ > 0; }

  TimeDelta& operator+=(const TimeDelta& other) noexcept;
  TimeDelta& operator-=(const TimeDelta& other) noexcept;
  TimeDelta operator+() const noexcept;
  TimeDelta operator-() const noexcept;
  TimeDelta& operator*=(const rational_t& mult) noexcept;
  TimeDelta& operator/=(const rational_t& div) noexcept;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  Slab slab_;
  rational_t fraction_;

  friend class Time;
};

// Time <cmp> Time
// clang-tidy: clang-tidy wants this removed in favor of friend
// declaration in different header.
bool operator==(const Time& a, const Time& b) noexcept;  // NOLINT
inline bool operator!=(const Time& a, const Time& b) noexcept {
  return not(a == b);
}
inline bool operator<(const Time& a, const Time& b) noexcept {
  // Non-equality test in second clause is required to avoid
  // assertions in Slab.
  return (a.slab() == b.slab() and a.fraction() < b.fraction()) or
         (a != b and a.slab() < b.slab());
}
inline bool operator>(const Time& a, const Time& b) noexcept {
  return b < a;
}
inline bool operator<=(const Time& a, const Time& b) noexcept {
  return not(a > b);
}
inline bool operator>=(const Time& a, const Time& b) noexcept {
  return not(a < b);
}

// TimeDelta <cmp> TimeDelta
inline bool operator==(const TimeDelta& a, const TimeDelta& b) noexcept {
  return a.slab() == b.slab() and a.fraction() == b.fraction();
}
inline bool operator!=(const TimeDelta& a, const TimeDelta& b) noexcept {
  return not(a == b);
}
inline bool operator<(const TimeDelta& a, const TimeDelta& b) noexcept {
  ASSERT(a.slab() == b.slab(),
         "Can't check cross-slab TimeDelta inequalities");
  return a.fraction() < b.fraction();
}
inline bool operator>(const TimeDelta& a, const TimeDelta& b) noexcept {
  return b < a;
}
inline bool operator<=(const TimeDelta& a, const TimeDelta& b) noexcept {
  return not(a > b);
}
inline bool operator>=(const TimeDelta& a, const TimeDelta& b) noexcept {
  return not(a < b);
}

// Time <op> Time
TimeDelta operator-(const Time& a, const Time& b) noexcept;

// Time <op> TimeDelta, TimeDelta <op> Time
inline Time operator+(Time a, const TimeDelta& b) noexcept {
  a += b;
  return a;
}

inline Time operator+(const TimeDelta& a, Time b) noexcept {
  b += a;
  return b;
}

inline Time operator-(Time a, const TimeDelta& b) noexcept {
  a -= b;
  return a;
}

// TimeDelta <op> TimeDelta
inline TimeDelta operator+(TimeDelta a, const TimeDelta& b) noexcept {
  a += b;
  return a;
}

inline TimeDelta operator-(TimeDelta a, const TimeDelta& b) noexcept {
  a -= b;
  return a;
}

// This returns a double rather than a rational so we can compare dt
// in different slabs.
double operator/(const TimeDelta& a, const TimeDelta& b) noexcept;

// rational <op> TimeDelta, TimeDelta <op> rational
inline TimeDelta operator*(TimeDelta a,
                           const TimeDelta::rational_t& b) noexcept {
  a *= b;
  return a;
}

inline TimeDelta operator*(const TimeDelta::rational_t& a,
                           TimeDelta b) noexcept {
  b *= a;
  return b;
}

inline TimeDelta operator/(TimeDelta a,
                           const TimeDelta::rational_t& b) noexcept {
  a /= b;
  return a;
}

inline TimeDelta abs(TimeDelta t) noexcept {
  if (not t.is_positive()) {
    t *= -1;
  }
  return t;
}

std::ostream& operator<<(std::ostream& os, const Time& t) noexcept;

std::ostream& operator<<(std::ostream& os, const TimeDelta& dt) noexcept;

// Time member functions
inline Time& Time::operator+=(const TimeDelta& delta) noexcept {
  *this = this->with_slab(delta.slab_);
  fraction_ += delta.fraction_;
  range_check();
  compute_value();
  return *this;
}

inline Time& Time::operator-=(const TimeDelta& delta) noexcept {
  *this = this->with_slab(delta.slab_);
  fraction_ -= delta.fraction_;
  range_check();
  compute_value();
  return *this;
}

// TimeDelta member functions
inline TimeDelta& TimeDelta::operator+=(const TimeDelta& other) noexcept {
  ASSERT(slab_ == other.slab_, "Can't add TimeDeltas from different slabs");
  fraction_ += other.fraction_;
  return *this;
}

inline TimeDelta& TimeDelta::operator-=(const TimeDelta& other) noexcept {
  ASSERT(slab_ == other.slab_,
         "Can't subtract TimeDeltas from different slabs");
  fraction_ -= other.fraction_;
  return *this;
}

inline TimeDelta TimeDelta::operator+() const noexcept { return *this; }

inline TimeDelta TimeDelta::operator-() const noexcept {
  return {slab_, -fraction_};
}

inline TimeDelta& TimeDelta::operator*=(const rational_t& mult) noexcept {
  fraction_ *= mult;
  return *this;
}

inline TimeDelta& TimeDelta::operator/=(const rational_t& div) noexcept {
  fraction_ /= div;
  return *this;
}

size_t hash_value(const Time& t) noexcept;

namespace std {
template <>
struct hash<Time> {
  size_t operator()(const Time& t) const noexcept;
};
}  // namespace std
