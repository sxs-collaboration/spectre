// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <functional>
#include <iosfwd>

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

template <typename T>
struct PrimitiveId {
  PrimitiveId() = default;
  PrimitiveId(const T& value) : value_(value){};
  operator T&() { return value_; }
  operator T() const { return value_; }

  // To support ObservationId
  double value() const noexcept { return static_cast<double>(value_); }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept { p | value_; };  // NOLINT

 private:
  T value_{};
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const PrimitiveId<T>& id) noexcept {
  return os << static_cast<T>(id);
}

template <typename T>
size_t hash_value(const PrimitiveId<T>& id) noexcept {
  size_t h = 0;
  boost::hash_combine(h, static_cast<T>(id));
  return h;
}

namespace std {
template <typename T>
struct hash<PrimitiveId<T>> {
  size_t operator()(const PrimitiveId<T>& id) const noexcept {
    return boost::hash<PrimitiveId<T>>{}(id);
  }
};
}  // namespace std
