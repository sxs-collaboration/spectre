// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <pup.h>

#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"

/// An optimized map with DirectionalId keys
template <size_t Dim, typename T>
class DirectionalIdMap : public FixedHashMap<maximum_number_of_neighbors(Dim),
                                             DirectionalId<Dim>, T> {
 public:
  using base =
      FixedHashMap<maximum_number_of_neighbors(Dim), DirectionalId<Dim>, T>;
  using base::base;
};

namespace PUP {
template <size_t Dim, typename T>
// NOLINTNEXTLINE(google-runtime-references)
void pup(PUP::er& p, DirectionalIdMap<Dim, T>& t) {
  pup(p, static_cast<typename DirectionalIdMap<Dim, T>::base&>(t));
}

template <size_t Dim, typename T>
// NOLINTNEXTLINE(google-runtime-references)
void operator|(PUP::er& p, DirectionalIdMap<Dim, T>& t) {
  p | static_cast<typename DirectionalIdMap<Dim, T>::base&>(t);
}
}  // namespace PUP
