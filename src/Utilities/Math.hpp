// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>

#include "Utilities/ForceInline.hpp"
#include "Utilities/TypeTraits.hpp"

/*!
 * \ingroup UtilitiesGroup
 * \brief Returns the number of digits in an integer number
 */
template <typename T>
SPECTRE_ALWAYS_INLINE T number_of_digits(const T number) {
  static_assert(tt::is_integer_v<std::decay_t<T>>,
                "Must call number_of_digits with an integer number");
  return number == 0 ? 1 : static_cast<decltype(number)>(
                               std::ceil(std::log10(std::abs(number) + 1)));
}
