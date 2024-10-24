// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <bit>

#if defined(__cpp_lib_int_pow2) && __cpp_lib_int_pow2 >= 202002L
// std::bit_floor and std::has_single_bit are defined in <bit>
#else
// std::bit_floor and std::has_single_bit are not defined in <bit> and
// therefore we provide the definitions
#include <cstdint>
#include <limits>

#include "Utilities/Requires.hpp"

// NOLINTNEXTLINE(cert-dcl58-cpp)
namespace std {
template <typename T,
          Requires<std::is_same_v<T, uint8_t> or std::is_same_v<T, uint16_t> or
                   std::is_same_v<T, uint32_t> or std::is_same_v<T, uint64_t> or
                   std::is_same_v<T, size_t> > = nullptr>
// NOLINTNEXTLINE(cert-dcl58-cpp)
constexpr T bit_floor(T x) noexcept {
  if (x != 0) {
    return T(1) << (std::numeric_limits<T>::digits - std::countl_zero(x) - 1);
  }
  return 0;
}

template <typename T,
          Requires<std::is_same_v<T, uint8_t> or std::is_same_v<T, uint16_t> or
                   std::is_same_v<T, uint32_t> or std::is_same_v<T, uint64_t> or
                   std::is_same_v<T, size_t> > = nullptr>
// NOLINTNEXTLINE(cert-dcl58-cpp)
constexpr bool has_single_bit(T x) noexcept {
  return std::popcount(x) == 1;
}
}  // namespace std
#endif
