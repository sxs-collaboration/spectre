// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// PUP routines for new C+11 STL containers and other standard
/// library objects Charm does not provide implementations for

#pragma once

#include <pup_stl.h>
#include <utility>

namespace PUP {

/// \ingroup ParallelGroup
/// Serialization of std::atomic for Charm++
template <typename T>
void pup(PUP::er& p, std::atomic<T>& d) noexcept {  // NOLINT
  T local_d;
  if(p.isUnpacking()) {
    p | local_d;
    d.store(std::move(local_d));
  } else {
    local_d = d.load();
    p | local_d;
  }
}

/// \ingroup ParallelGroup
/// Serialization of std::atomic for Charm++
template <typename T>
void operator|(er& p, std::atomic<T>& d) noexcept {  // NOLINT
  pup(p, d);
}
}  // namespace PUP
