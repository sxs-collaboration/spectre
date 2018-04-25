// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/Time.hpp"

#include <boost/functional/hash.hpp>
#include <ostream>
#include <pup.h>

// Time implementations

void Time::pup(PUP::er& p) noexcept {
  p | slab_;
  p | fraction_;
  p | value_;
}

Time Time::with_slab(const Slab& new_slab) const noexcept {
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

void Time::compute_value() noexcept {
  if (is_at_slab_end()) {
    // Protection against rounding error.
    value_ = slab_.end_;
  } else {
    value_ = slab_.start_ + (slab_.duration() * fraction_).value();
  }
}

bool operator==(const Time& a, const Time& b) noexcept {
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

TimeDelta operator-(const Time& a, const Time& b) noexcept {
  if (a.slab() == b.slab()) {
    return {a.slab(), a.fraction() - b.fraction()};
  } else if (a.slab().is_followed_by(b.slab())) {
    if (a.is_at_slab_end()) {
      return {b.slab(), -b.fraction()};
    } else {
      ASSERT(b.is_at_slab_start(),
             "Can't subtract times from different slabs");
      return {a.slab(), a.fraction() - 1};
    }
  } else {
    ASSERT(a.slab().is_preceeded_by(b.slab()),
           "Can't subtract times from different slabs");
    if (a.is_at_slab_start()) {
      return {b.slab(), 1 - b.fraction()};
    } else {
      ASSERT(b.is_at_slab_end(),
             "Can't subtract times from different slabs");
      return {a.slab(), a.fraction()};
    }
  }
}

std::ostream& operator<<(std::ostream& os, const Time& t) noexcept {
  return os << t.slab() << ":" << t.fraction();
}

// TimeDelta implementations

void TimeDelta::pup(PUP::er& p) noexcept {
  p | slab_;
  p | fraction_;
}

double operator/(const TimeDelta& a, const TimeDelta& b) noexcept {
  // We need to use double/double here because this is the
  // implementation of TimeDelta/TimeDelta.
  return (a.fraction() / b.fraction()).value() *
         (a.slab().duration().value() / b.slab().duration().value());
}

std::ostream& operator<<(std::ostream& os, const TimeDelta& dt) noexcept {
  return os << dt.slab() << ":" << dt.fraction();
}

size_t hash_value(const Time& t) noexcept {
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
size_t hash<Time>::operator()(const Time& t) const noexcept {
  return boost::hash<Time>{}(t);
}
}  // namespace std
