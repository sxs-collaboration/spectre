// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <type_traits>
#include <vector>

#include "Utilities/ForceInline.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/IsA.hpp"
#include "Utilities/TypeTraits/IsInteger.hpp"

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
 * \tparam CoeffsIterable The type of the polynomial coefficients \p coeffs. Can
 * be a `std::vector<double>` or `std::array<double>`, which means the
 * coefficients are constant for all values in \p x. Each coefficient can also
 * be a vector type of typically the same size as \p x, which means the
 * coefficients vary with the elements in \p x.
 * \tparam DataType The type of the polynomial variable \p x. Must support
 * `make_with_value<DataType, DataType>`, as well as (elementwise) addition with
 * `CoeffsIterable::value_type` and multiplication with `DataType`.
 */
template <typename CoeffsIterable, typename DataType>
DataType evaluate_polynomial(const CoeffsIterable& coeffs,
                             const DataType& x) noexcept {
  return std::accumulate(
      coeffs.rbegin(), coeffs.rend(), make_with_value<DataType>(x, 0.),
      [&x](const DataType& state, const auto& element) noexcept {
        return state * x + element;
      });
}

/// \ingroup UtilitiesGroup

/// \brief Defines the Heaviside step function \f$\Theta\f$ for arithmetic
/// types.  \f$\Theta(0) = 1\f$.
template <typename T, Requires<std::is_arithmetic<T>::value> = nullptr>
constexpr T step_function(const T& arg) noexcept {
  return static_cast<T>((arg >= static_cast<T>(0)) ? 1 : 0);
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Smoothly interpolates from 0 to 1 between `lower_edge` and
 * `upper_edge` with a Hermite polynomial of degree `2 * N + 1`.
 *
 * The smoothstep function is
 *
 * \f{align*}
 * S_N(x) = \begin{cases}
 * 0 &\quad \text{for} \quad x\leq x_0 \\
 * \tilde{S}_N((x - x_0) / (x_1 - x_0))
 * &\quad \text{for} \quad x_0 \leq x\leq x_1 \\
 * 1 &\quad \text{for} \quad x_1\leq x \\
 * \end{cases}
 * \f}
 *
 * where \f$x_0\f$ is `lower_edge`, \f$x_1\f$ is `upper_edge`, and, up to
 * \f$N=3\f$,
 *
 * \f{align*}
 * \tilde{S}_0(x) &= x \\
 * \tilde{S}_1(x) &= 3x^2 - 2x^3 \\
 * \tilde{S}_2(x) &= 10x^3 - 15x^4 + 6x^5 \\
 * \tilde{S}_3(x) &= 35x^4 - 84x^5 + 70x^6 - 20x^7
 * \text{.}
 * \f}
 */
template <size_t N, typename DataType>
DataType smoothstep(const double lower_edge, const double upper_edge,
                    const DataType& arg) noexcept {
  ASSERT(lower_edge < upper_edge,
         "Requires lower_edge < upper_edge, but lower_edge="
             << lower_edge << " and upper_edge=" << upper_edge);
  using std::clamp;
  return evaluate_polynomial(
      []() noexcept -> std::array<double, 2 * N + 2> {
        static_assert(N <= 3,
                      "The smoothstep function is currently only implemented "
                      "for N <= 3.");
        if constexpr (N == 0) {
          return {0., 1};
        } else if constexpr (N == 1) {
          return {0., 0., 3., -2};
        } else if constexpr (N == 2) {
          return {0., 0., 0., 10., -15., 6.};
        } else if constexpr (N == 3) {
          return {0., 0., 0., 0., 35., -84., 70., -20.};
        }
      }(),
      static_cast<DataType>(
          clamp((arg - lower_edge) / (upper_edge - lower_edge), 0., 1.)));
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
