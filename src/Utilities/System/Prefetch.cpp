// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/System/Prefetch.hpp"

#include <ostream>
#include <stdexcept>

namespace sys {
std::ostream& operator<<(std::ostream& os, const PrefetchTo cache_location) {
  switch (cache_location) {
    case PrefetchTo::L1Cache:
      return os << "L1Cache";
    case PrefetchTo::L2Cache:
      return os << "L2Cache";
    case PrefetchTo::L3Cache:
      return os << "L3Cache";
    case PrefetchTo::NonTemporal:
      return os << "NonTemporal";
    case PrefetchTo::WriteL1Cache:
      return os << "WriteL1Cache";
    case PrefetchTo::WriteL2Cache:
      return os << "WriteL2Cache";
    default:
      throw std::runtime_error("Unknown value of cache_location");
  };
}
}  // namespace sys
