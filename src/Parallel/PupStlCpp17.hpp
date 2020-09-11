// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <pup_stl.h>
#include <utility>

namespace PUP {
/// \ingroup ParallelGroup
/// Serialization of std::optional for Charm++
template <typename T>
void pup(er& p, std::optional<T>& t) noexcept {  // NOLINT
  bool valid = false;
  if (p.isUnpacking()) {
    p | valid;
    if (valid) {
      T value{};
      p | value;
      t = std::move(value);
    } else {
      t.reset();
    }
  } else {
    valid = t.has_value();
    p | valid;
    if (valid) {
      p | *t;
    }
  }
}

/// \ingroup ParallelGroup
/// Serialization of std::optional for Charm++
template <typename T>
void operator|(er& p, std::optional<T>& t) noexcept {  // NOLINT
  pup(p, t);
}
}  // namespace PUP
