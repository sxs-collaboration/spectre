// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <numeric>
#include <vector>

#include "Utilities/ForceInline.hpp"
#include "Utilities/MakeWithValue.hpp"
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

/*!
 * \ingroup UtilitiesGroup
 * \brief Evaluate a polynomial \f$\sum_{p=0}^N c_p x^p\f$ with Horner's rule
 *
 * \param coeffs The polynomial coefficients \f$c_p\f$ ordered from constant to
 * largest power
 * \param x The polynomial variable \f$x\f$
 *
 * \tparam U The type of the polynomial coefficients \p coeffs. Can be `double`,
 * which means the coefficients are constant for all values in \p x. Can also be
 * a vector type of typically the same size as `T`, which means the coefficients
 * vary with the elements in \p x.
 * \tparam T The type of the polynomial variable \p x. Must support
 * `make_with_value<T, T>`, as well as (elementwise) addition with `U` and
 * multiplication with `T`.
 */
template <typename U, typename T>
T evaluate_polynomial(const std::vector<U>& coeffs, const T& x) noexcept {
  return std::accumulate(
      coeffs.rbegin(), coeffs.rend(), make_with_value<T>(x, 0.),
      [&x](const T& state, const U& element) { return state * x + element; });
}
