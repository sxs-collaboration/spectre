// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/Slab.hpp"

#include <boost/functional/hash.hpp>
#include <ostream>
#include <pup.h>

#include "Time/Time.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

Slab::Slab(const double start, const double end) : start_(start), end_(end) {
  ASSERT(start_ < end_, "Backwards Slab: " << start_ << " >= " << end_);
}

Slab Slab::with_duration_from_start(const double start, const double duration) {
  return {start, start + duration};
}

Slab Slab::with_duration_to_end(const double end, const double duration) {
  return {end - duration, end};
}

Time Slab::start() const { return {*this, 0}; }

Time Slab::end() const { return {*this, 1}; }

TimeDelta Slab::duration() const { return {*this, 1}; }

Slab Slab::advance() const { return {end_, end_ + (end_ - start_)}; }

Slab Slab::retreat() const { return {start_ - (end_ - start_), start_}; }

Slab Slab::advance_towards(const TimeDelta& dt) const {
  ASSERT(dt.is_positive() or (-dt).is_positive(),
         "Can't advance along a zero time vector");
  return dt.is_positive() ? advance() : retreat();
}

Slab Slab::with_duration_from_start(const double duration) const {
  return {start_, start_ + duration};
}

Slab Slab::with_duration_to_end(const double duration) const {
  return {end_ - duration, end_};
}

bool Slab::is_followed_by(const Slab& other) const {
  return end_ == other.start_;
}

bool Slab::is_preceeded_by(const Slab& other) const {
  return other.is_followed_by(*this);
}

bool Slab::overlaps(const Slab& other) const {
  return not(end_ <= other.start_ or start_ >= other.end_);
}

void Slab::pup(PUP::er& p) {
  p | start_;
  p | end_;
}

bool operator==(const Slab& a, const Slab& b) {
  return a.start_ == b.start_ and a.end_ == b.end_;
}

bool operator!=(const Slab& a, const Slab& b) { return not(a == b); }

bool operator<(const Slab& a, const Slab& b) {
  ASSERT(a == b or a.end_ <= b.start_ or a.start_ >= b.end_,
         "Cannot compare overlapping slabs " << a << " and " << b);
  return a.end_ <= b.start_;
}

bool operator>(const Slab& a, const Slab& b) { return b < a; }

bool operator<=(const Slab& a, const Slab& b) { return not(a > b); }

bool operator>=(const Slab& a, const Slab& b) { return not(a < b); }

std::ostream& operator<<(std::ostream& os, const Slab& s) {
  return os << "Slab[" << s.start().value() << "," << s.end().value() << "]";
}

size_t hash_value(const Slab& s) {
  size_t h = 0;
  boost::hash_combine(h, s.start().value());
  boost::hash_combine(h, s.end().value());
  return h;
}

// clang-tidy: do not modify std namespace (okay for hash)
namespace std {  // NOLINT
size_t hash<Slab>::operator()(const Slab& s) const {
  return boost::hash<Slab>{}(s);
}
}  // namespace std
