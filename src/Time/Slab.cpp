// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/Slab.hpp"

#include <boost/functional/hash.hpp>
#include <ostream>
#include <pup.h>

#include "Time/Time.hpp"

// These things are here to avoid problems with Time possibly being
// incomplete in the header.

Time Slab::start() const noexcept { return {*this, 0}; }

Time Slab::end() const noexcept { return {*this, 1}; }

TimeDelta Slab::duration() const noexcept { return {*this, 1}; }

Slab Slab::advance_towards(const TimeDelta& dt) const noexcept {
  ASSERT(dt.is_positive() or (-dt).is_positive(),
         "Can't advance along a zero time vector");
  return dt.is_positive() ? advance() : retreat();
}

void Slab::pup(PUP::er& p) noexcept {
  p | start_;
  p | end_;
}

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
