// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/Time.hpp"

#include <boost/functional/hash.hpp>
#include <cmath>
#include <ostream>
#include <pup.h>
#include <utility>

#include "Utilities/ErrorHandling/Assert.hpp"

// Time implementations

Time::Time(Slab slab, rational_t fraction)
    // clang-tidy: move trivially copyable type
    : slab_(std::move(slab)), fraction_(std::move(fraction)) {  // NOLINT
  range_check();
  compute_value();
}

Time Time::with_slab(const Slab& new_slab) const {
  if (new_slab == slab_) {
    return *this;
  } else if (is_at_slab_start()) {
    if (slab_.start_ == new_slab.start_) {
      return new_slab.start();
    } else {
      ASSERT(slab_.start() == new_slab.end(),
             "Can't move " << fraction_ << " " << slab_ << " to slab "
             << new_slab);
      return new_slab.end();
    }
  } else {
    ASSERT(is_at_slab_end(), "Can't move " << fraction_ << " " << slab_
                                           << " to slab " << new_slab);
    if (slab_.end_ == new_slab.end_) {
      return new_slab.end();
    } else {
      ASSERT(slab_.end() == new_slab.start(),
             "Can't move " << fraction_ << " " << slab_ << " to slab "
             << new_slab);
      return new_slab.start();
    }
  }
}

Time& Time::operator+=(const TimeDelta& delta) {
  *this = this->with_slab(delta.slab_);
  fraction_ += delta.fraction_;
  range_check();
  compute_value();
  return *this;
}

Time& Time::operator-=(const TimeDelta& delta) {
  *this = this->with_slab(delta.slab_);
  fraction_ -= delta.fraction_;
  range_check();
  compute_value();
  return *this;
}

bool Time::is_at_slab_start() const { return fraction_ == 0; }
bool Time::is_at_slab_end() const { return fraction_ == 1; }
bool Time::is_at_slab_boundary() const {
  return is_at_slab_start() or is_at_slab_end();
}

void Time::pup(PUP::er& p) {
  p | slab_;
  p | fraction_;
  p | value_;
}

bool Time::StructuralCompare::operator()(const Time& a, const Time& b) const {
  if (a.fraction().numerator() != b.fraction().numerator()) {
    return a.fraction().numerator() < b.fraction().numerator();
  }
  if (a.fraction().denominator() != b.fraction().denominator()) {
    return a.fraction().denominator() < b.fraction().denominator();
  }
  return a.slab() < b.slab();
}

void Time::compute_value() {
  if (is_at_slab_end()) {
    // Protection against rounding error.
    value_ = slab_.end_;
  } else {
    value_ = slab_.start_ + (slab_.duration() * fraction_).value();
  }
}

void Time::range_check() const {
  ASSERT(fraction_ >= 0 and fraction_ <= 1,
         "Out of range slab fraction: " << fraction_);
}

// TimeDelta implementations

TimeDelta::TimeDelta(Slab slab, rational_t fraction)
    // clang-tidy: move trivially copyable type
    : slab_(std::move(slab)), fraction_(std::move(fraction)) {}  // NOLINT

TimeDelta TimeDelta::with_slab(const Slab& new_slab) const {
  return {new_slab, fraction()};
}

double TimeDelta::value() const {
  return (slab_.end_ - slab_.start_) * fraction_.value();
}

bool TimeDelta::is_positive() const { return fraction_ > 0; }

TimeDelta& TimeDelta::operator+=(const TimeDelta& other) {
  ASSERT(slab_ == other.slab_, "Can't add TimeDeltas from different slabs");
  fraction_ += other.fraction_;
  return *this;
}

TimeDelta& TimeDelta::operator-=(const TimeDelta& other) {
  ASSERT(slab_ == other.slab_,
         "Can't subtract TimeDeltas from different slabs");
  fraction_ -= other.fraction_;
  return *this;
}

TimeDelta TimeDelta::operator+() const { return *this; }

TimeDelta TimeDelta::operator-() const { return {slab_, -fraction_}; }

TimeDelta& TimeDelta::operator*=(const rational_t& mult) {
  fraction_ *= mult;
  return *this;
}

TimeDelta& TimeDelta::operator/=(const rational_t& div) {
  fraction_ /= div;
  return *this;
}

void TimeDelta::pup(PUP::er& p) {
  p | slab_;
  p | fraction_;
}

// Free function implementations

// Time <cmp> Time

bool operator==(const Time& a, const Time& b) {
  if (a.slab() == b.slab()) {
    return a.fraction() == b.fraction();
  } else {
    return (a.slab().is_followed_by(b.slab()) and
            a.is_at_slab_end() and
            b.is_at_slab_start()) or
           (a.slab().is_preceeded_by(b.slab()) and
            a.is_at_slab_start() and
            b.is_at_slab_end()) or
           (a.slab().start_ == b.slab().start_ and
            a.is_at_slab_start() and
            b.is_at_slab_start()) or
           (a.slab().end_ == b.slab().end_ and
            a.is_at_slab_end() and
            b.is_at_slab_end());
  }
}

bool operator!=(const Time& a, const Time& b) { return not(a == b); }

bool operator<(const Time& a, const Time& b) {
  if (a.slab() == b.slab()) {
    return a.fraction() < b.fraction();
  }
  if (not a.slab().overlaps(b.slab())) {
    return a != b and a.slab() < b.slab();
  }
  if ((a.slab().start() == b.slab().start() and a.is_at_slab_start()) or
      (a.slab().end() == b.slab().end() and a.is_at_slab_end())) {
    return a.with_slab(b.slab()) < b;
  } else {
    return a < b.with_slab(a.slab());
  }
}

bool operator>(const Time& a, const Time& b) { return b < a; }

bool operator<=(const Time& a, const Time& b) { return not(a > b); }

bool operator>=(const Time& a, const Time& b) { return not(a < b); }

// TimeDelta <cmp> TimeDelta

bool operator==(const TimeDelta& a, const TimeDelta& b) {
  return a.slab() == b.slab() and a.fraction() == b.fraction();
}

bool operator!=(const TimeDelta& a, const TimeDelta& b) { return not(a == b); }

bool operator<(const TimeDelta& a, const TimeDelta& b) {
  ASSERT(a.slab() == b.slab(), "Can't check cross-slab TimeDelta inequalities");
  return a.fraction() < b.fraction();
}

bool operator>(const TimeDelta& a, const TimeDelta& b) { return b < a; }

bool operator<=(const TimeDelta& a, const TimeDelta& b) { return not(a > b); }

bool operator>=(const TimeDelta& a, const TimeDelta& b) { return not(a < b); }

// Time <op> Time

TimeDelta operator-(const Time& a, const Time& b) {
  if (a.slab() == b.slab()) {
    return {a.slab(), a.fraction() - b.fraction()};
  } else if (a.slab().is_followed_by(b.slab())) {
    if (a.is_at_slab_end()) {
      return {b.slab(), -b.fraction()};
    } else {
      ASSERT(b.is_at_slab_start(),
             "Can't subtract times from different slabs: " << a << " - " << b);
      return {a.slab(), a.fraction() - 1};
    }
  } else {
    ASSERT(a.slab().is_preceeded_by(b.slab()),
           "Can't subtract times from different slabs: " << a << " - " << b);
    if (a.is_at_slab_start()) {
      return {b.slab(), 1 - b.fraction()};
    } else {
      ASSERT(b.is_at_slab_end(),
             "Can't subtract times from different slabs: " << a << " - " << b);
      return {a.slab(), a.fraction()};
    }
  }
}

// Time <op> TimeDelta, TimeDelta <op> Time
Time operator+(Time a, const TimeDelta& b) {
  a += b;
  return a;
}

Time operator+(const TimeDelta& a, Time b) {
  b += a;
  return b;
}

Time operator-(Time a, const TimeDelta& b) {
  a -= b;
  return a;
}

// TimeDelta <op> TimeDelta

TimeDelta operator+(TimeDelta a, const TimeDelta& b) {
  a += b;
  return a;
}

TimeDelta operator-(TimeDelta a, const TimeDelta& b) {
  a -= b;
  return a;
}

double operator/(const TimeDelta& a, const TimeDelta& b) {
  // We need to use double/double here because this is the
  // implementation of TimeDelta/TimeDelta.
  return (a.fraction() / b.fraction()).value() *
         (a.slab().duration().value() / b.slab().duration().value());
}

// rational <op> TimeDelta, TimeDelta <op> rational
TimeDelta operator*(TimeDelta a, const TimeDelta::rational_t& b) {
  a *= b;
  return a;
}

TimeDelta operator*(const TimeDelta::rational_t& a, TimeDelta b) {
  b *= a;
  return b;
}

TimeDelta operator/(TimeDelta a, const TimeDelta::rational_t& b) {
  a /= b;
  return a;
}

// Miscellaneous other functions for Time

std::ostream& operator<<(std::ostream& os, const Time& t) {
  return os << t.slab() << ":" << t.fraction();
}

size_t hash_value(const Time& t) {
  // We need equal Times to have equal hashes.  Times at matching ends
  // of slabs are equal, so we can't use the slab and fraction in that
  // case.
  if (t.is_at_slab_boundary()) {
    return boost::hash<double>{}(t.value());
  } else {
    size_t h = 0;
    boost::hash_combine(h, t.slab());
    boost::hash_combine(h, t.fraction());
    return h;
  }
}

// clang-tidy: do not modify std namespace (okay for hash)
namespace std {  // NOLINT
size_t hash<Time>::operator()(const Time& t) const {
  return boost::hash<Time>{}(t);
}
}  // namespace std

// Miscellaneous other functions for TimeDelta

TimeDelta abs(TimeDelta t) {
  if (not t.is_positive()) {
    t *= -1;
  }
  return t;
}

std::ostream& operator<<(std::ostream& os, const TimeDelta& dt) {
  return os << dt.slab() << ":" << dt.fraction();
}
