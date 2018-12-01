// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

// @{
/// \ingroup UtilitiesGroup
/// \brief Returns an appropriate signaling NaN for fundamantal or multi-field
/// types (such as `std::complex`).
template <typename T>
T make_signaling_NaN(const T& /*meta*/) noexcept {
  return std::numeric_limits<T>::signaling_NaN();
}

template <typename T>
std::complex<T> make_signaling_NaN(const std::complex<T>& /*meta*/) noexcept {
  return {std::numeric_limits<T>::signaling_NaN(),
          std::numeric_limits<T>::signaling_NaN()};
}

template <typename T>
T make_signaling_NaN() noexcept {
  return make_signaling_NaN(static_cast<T>(0));
}
// @}
