// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/Slab.hpp"

#include <boost/functional/hash.hpp>
#include <ostream>
#include <pup.h>

#include "Time/Time.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

Slab::Slab(const double start, const double end) noexcept
    : start_(start), end_(end) {
  ASSERT(start_ < end_, "Backwards Slab: " << start_ << " >= " << end_);
}

Slab Slab::with_duration_from_start(const double start,
                                    const double duration) noexcept {
  return {start, start + duration};
}

Slab Slab::with_duration_to_end(const double end,
                                const double duration) noexcept {
  return {end - duration, end};
}

Time Slab::start() const noexcept { return {*this, 0}; }

Time Slab::end() const noexcept { return {*this, 1}; }

TimeDelta Slab::duration() const noexcept { return {*this, 1}; }

Slab Slab::advance() const noexcept { return {end_, end_ + (end_ - start_)}; }

Slab Slab::retreat() const noexcept {
  return {start_ - (end_ - start_), start_};
}

Slab Slab::advance_towards(const TimeDelta& dt) const noexcept {
  ASSERT(dt.is_positive() or (-dt).is_positive(),
         "Can't advance along a zero time vector");
  return dt.is_positive() ? advance() : retreat();
}

Slab Slab::with_duration_from_start(const double duration) const noexcept {
  return {start_, start_ + duration};
}

Slab Slab::with_duration_to_end(const double duration) const noexcept {
  return {end_ - duration, end_};
}

bool Slab::is_followed_by(const Slab& other) const noexcept {
  return end_ == other.start_;
}

bool Slab::is_preceeded_by(const Slab& other) const noexcept {
  return other.is_followed_by(*this);
}

bool Slab::overlaps(const Slab& other) const noexcept {
  return not(end_ <= other.start_ or start_ >= other.end_);
}

void Slab::pup(PUP::er& p) noexcept {
  p | start_;
  p | end_;
}

bool operator==(const Slab& a, const Slab& b) noexcept {
  return a.start_ == b.start_ and a.end_ == b.end_;
}

bool operator!=(const Slab& a, const Slab& b) noexcept { return not(a == b); }

bool operator<(const Slab& a, const Slab& b) noexcept {
  ASSERT(a == b or a.end_ <= b.start_ or a.start_ >= b.end_,
         "Cannot compare overlapping slabs " << a << " and " << b);
  return a.end_ <= b.start_;
}

bool operator>(const Slab& a, const Slab& b) noexcept { return b < a; }

bool operator<=(const Slab& a, const Slab& b) noexcept { return not(a > b); }

bool operator>=(const Slab& a, const Slab& b) noexcept { return not(a < b); }

std::ostream& operator<<(std::ostream& os, const Slab& s) noexcept {
  return os << "Slab[" << s.start().value() << "," << s.end().value() << "]";
}

size_t hash_value(const Slab& s) noexcept {
  size_t h = 0;
  boost::hash_combine(h, s.start().value());
  boost::hash_combine(h, s.end().value());
  return h;
}

// clang-tidy: do not modify std namespace (okay for hash)
namespace std {  // NOLINT
size_t hash<Slab>::operator()(const Slab& s) const noexcept {
  return boost::hash<Slab>{}(s);
}
}  // namespace std
