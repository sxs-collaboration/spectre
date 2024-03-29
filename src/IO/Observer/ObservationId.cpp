// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Observer/ObservationId.hpp"

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <utility>

namespace observers {
ObservationKey::ObservationKey(std::string tag)
    : key_(std::hash<std::string>{}(tag)), tag_(std::move(tag)) {}

void ObservationKey::pup(PUP::er& p) {
  p | key_;
  p | tag_;
}

const std::string& ObservationKey::tag() const { return tag_; }

bool operator==(const ObservationKey& lhs, const ObservationKey& rhs) {
  return lhs.key() == rhs.key() and lhs.tag() == rhs.tag();
}

bool operator!=(const ObservationKey& lhs, const ObservationKey& rhs) {
  return not(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os, const ObservationKey& t) {
  return os << '(' << t.tag() << ')';
}

ObservationId::ObservationId(const double t, std::string tag)
    : observation_key_(std::move(tag)),
      combined_hash_([&t](size_t type_hash) {
        size_t combined = type_hash;
        boost::hash_combine(combined, t);
        return combined;
      }(observation_key_.key())),
      value_(t) {}

void ObservationId::pup(PUP::er& p) {
  p | observation_key_;
  p | combined_hash_;
  p | value_;
}

bool operator==(const ObservationId& lhs, const ObservationId& rhs) {
  return lhs.hash() == rhs.hash() and lhs.value() == rhs.value();
}

bool operator!=(const ObservationId& lhs, const ObservationId& rhs) {
  return not(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os, const ObservationId& t) {
  return os << '(' << t.observation_key() << "," << t.hash() << ',' << t.value()
            << ')';
}
}  // namespace observers
