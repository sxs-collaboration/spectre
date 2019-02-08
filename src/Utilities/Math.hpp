// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <numeric>
#include <type_traits>
#include <vector>

#include "Utilities/ForceInline.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

// using for overload resolution with blaze
// clang-tidy doesn't want these in the global namespace
using std::conj; //NOLINT
using std::imag; //NOLINT
using std::real; //NOLINT

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

/// \ingroup UtilitiesGroup

/// \brief Defines the Heaviside step function \f$\Theta\f$ for arithmetic
/// types.  \f$\Theta(0) = 1\f$.
template <typename T, Requires<std::is_arithmetic<T>::value> = nullptr>
constexpr T step_function(const T& arg) noexcept {
  return static_cast<T>((arg >= static_cast<T>(0)) ? 1 : 0);
}

/// \ingroup UtilitiesGroup
/// \brief Defines the inverse square-root (\f$1/\sqrt{x}\f$) for arithmetic
/// and complex types
template <typename T, Requires<std::is_arithmetic<T>::value or
                               tt::is_a_v<std::complex, T>> = nullptr>
auto invsqrt(const T& arg) noexcept {
  return static_cast<T>(1.0) / sqrt(arg);
}

/// \ingroup UtilitiesGroup
/// \brief Defines the inverse cube-root (\f$1/\sqrt[3]{x}\f$) for arithmetic
/// types
template <typename T, Requires<std::is_arithmetic<T>::value> = nullptr>
auto invcbrt(const T& arg) noexcept {
  return static_cast<T>(1.0) / cbrt(arg);
}

namespace sgn_detail {
template <typename T>
constexpr T sgn(const T& val, std::true_type /*is_signed*/) noexcept {
  return static_cast<T>(static_cast<T>(0) < val) -
         static_cast<T>(val < static_cast<T>(0));
}

template <typename T>
constexpr T sgn(const T& val, std::false_type /*is_signed*/) noexcept {
  return static_cast<T>(static_cast<T>(0) < val);
}
}  // namespace sgn_detail

/// \ingroup UtilitiesGroup
/// \brief Compute the sign function of `val` defined as `1` if `val > 0`, `0`
/// if `val == 0`, and `-1` if `val < 0`.
template <typename T>
constexpr T sgn(const T& val) noexcept {
  return sgn_detail::sgn(val, std::is_signed<T>{});
}
