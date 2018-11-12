// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Direction.hpp"

/// \ingroup DataStructuresGroup
/// \ingroup ComputationalDomainGroup
/// An optimized map with Direction keys
template <size_t Dim, typename T>
class DirectionMap
    : public FixedHashMap<2 * Dim, Direction<Dim>, T, DirectionHash<Dim>> {
 public:
  using base = FixedHashMap<2 * Dim, Direction<Dim>, T, DirectionHash<Dim>>;
  using base::base;
};

namespace PUP {
template <size_t Dim, typename T>
// NOLINTNEXTLINE(google-runtime-references)
void pup(PUP::er& p, DirectionMap<Dim, T>& t) noexcept {
  pup(p, static_cast<typename DirectionMap<Dim, T>::base&>(t));
}

template <size_t Dim, typename T>
// NOLINTNEXTLINE(google-runtime-references)
void operator|(PUP::er& p, DirectionMap<Dim, T>& t) noexcept {
  p | static_cast<typename DirectionMap<Dim, T>::base&>(t);
}
}  // namespace PUP
