// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

namespace sys {
/// The cache location to prefetch data to.
enum class PrefetchTo : int {
  /// Prefetch into L1 data cache
  ///
  /// Typically the fastest CPU cache, about 32 KB in size.
  L1Cache = 3,
  /// Prefetch into L2 data cache
  ///
  /// Typically the second fastest CPU cache, about 128 or 256 KB in size.
  L2Cache = 2,
  /// Prefetch into L3 data cache
  ///
  /// Typically the slowest CPU cache and is shared among multiple cores with
  /// sizes varying by factors of several.
  L3Cache = 1,
  /// Non-temporal is an element that is unlikely to be re-used. E.g., read
  /// once, written but never read.
  NonTemporal = 0,
  /// Prefetch to L1 data cache for writing
  WriteL1Cache = 7,
  /// Prefetch to L2 data cache for writing
  WriteL2Cache = 6
};

/// \brief Prefetch data into a specific level of data cache.
template <PrefetchTo CacheLocation>
#if defined(__GNUC__)
__attribute__((always_inline)) inline
#endif
void prefetch(const void* address_to_prefetch) {
  // The enum values are bit flags where the lowest two bits (right-most)
  // control the cache level and the 3rd bit controls whether it is a write-only
  // or read-write operation.
  __builtin_prefetch(address_to_prefetch,
                     (static_cast<int>(CacheLocation) >> 2) & 1,
                     static_cast<int>(CacheLocation) & 0x3);
}

std::ostream& operator<<(std::ostream& os, PrefetchTo cache_location);
}  // namespace sys
