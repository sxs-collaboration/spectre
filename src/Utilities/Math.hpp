// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <type_traits>
#include <vector>

#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/IsA.hpp"
#include "Utilities/TypeTraits/IsInteger.hpp"

// using for overload resolution with blaze
// clang-tidy doesn't want these in the global namespace
using std::conj;  // NOLINT
using std::imag;  // NOLINT
using std::real;  // NOLINT

/*!
 * \ingroup UtilitiesGroup
 * \brief Returns the number of digits in an integer number
 */
template <typename T>
SPECTRE_ALWAYS_INLINE T number_of_digits(const T number) {
  static_assert(tt::is_integer_v<std::decay_t<T>>,
                "Must call number_of_digits with an integer number");
  return number == 0 ? 1
                     : static_cast<decltype(number)>(
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
DataType evaluate_polynomial(const CoeffsIterable& coeffs, const DataType& x) {
  return std::accumulate(coeffs.rbegin(), coeffs.rend(),
                         make_with_value<DataType>(x, 0.),
                         [&x](const DataType& state, const auto& element) {
                           return state * x + element;
                         });
}

/// \ingroup UtilitiesGroup

/// \brief Defines the Heaviside step function \f$\Theta\f$ for arithmetic
/// types.  \f$\Theta(0) = 1\f$.
template <typename T, Requires<std::is_arithmetic<T>::value> = nullptr>
constexpr T step_function(const T& arg) {
  return static_cast<T>((arg >= static_cast<T>(0)) ? 1 : 0);
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Smoothly interpolates from 0 to 1 between `lower_edge` and
 * `upper_edge` with Hermite interpolation of polynomial degree `2 * N + 1`.
 *
 * The smoothstep function is
 *
 * \begin{align*}
 * S_N(x) = \begin{cases}
 * 0 &\quad \text{for} \quad x\leq x_0 \\
 * \tilde{S}_N((x - x_0) / (x_1 - x_0))
 * &\quad \text{for} \quad x_0 \leq x\leq x_1 \\
 * 1 &\quad \text{for} \quad x_1\leq x \\
 * \end{cases}
 * \end{align*}
 *
 * where \f$x_0\f$ is `lower_edge` and \f$x_1\f$ is `upper_edge`. The general
 * form of the polynomial \f$\tilde{S}_N(x)\f$ is:
 *
 * \begin{equation}
 * \tilde{S}_N(x) = x^{N+1} \sum_{k=0}^{N} \binom{N+k}{k}
 *   \binom{2N+1}{N-k} (-x)^k
 * \end{equation}
 *
 * The first few polynomials are:
 *
 * \begin{align*}
 * \tilde{S}_0(x) &= x \\
 * \tilde{S}_1(x) &= 3x^2 - 2x^3 \\
 * \tilde{S}_2(x) &= 10x^3 - 15x^4 + 6x^5 \\
 * \tilde{S}_3(x) &= 35x^4 - 84x^5 + 70x^6 - 20x^7
 * \text{.}
 * \end{align*}
 *
 * This function is $C^N$ continuous at the edges, i.e. the first $N$
 * derivatives are continuous at the edges.
 *
 * If `lower_edge` and `upper_edge` are equal, this function reduces to the
 * step function $\Theta(x - x_0)$.
 */
template <size_t N, typename DataType>
DataType smoothstep(const double lower_edge, const double upper_edge,
                    const DataType& arg) {
  if (lower_edge == upper_edge) {
    return step_function(arg - lower_edge);
  }
  ASSERT(lower_edge < upper_edge,
         "Requires lower_edge < upper_edge, but lower_edge="
             << lower_edge << " and upper_edge=" << upper_edge);
  constexpr auto coeffs = []() {
    std::array<double, 2 * N + 2> result{};
    for (size_t k = 0; k <= N; ++k) {
      result[N + 1 + k] = (k % 2 == 0 ? 1. : -1.) * binomial(2 * N + 1, N - k) *
                          binomial(N + k, k);
    }
    return result;
  }();
  using std::clamp;
  const DataType x = clamp(
      static_cast<DataType>((arg - lower_edge) / (upper_edge - lower_edge)), 0.,
      1.);
  return evaluate_polynomial(coeffs, x);
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Derivative of the $N$-th order smoothstep function
 *
 * The general form of the derivative of the smoothstep function is:
 *
 * \begin{equation}
 * \tilde{S}_N'(x) = (2N + 1) \binom{2N}{N} (x - x^2)^N
 * \end{equation}
 *
 * Since the smoothstep function is $C^N$ continuous at the edges, this
 * function is $C^{N-1}$ continuous at the edges.
 *
 * If `lower_edge` and `upper_edge` are equal, this function reduces to the
 * derivative of the step function, which is zero everywhere except at the edge
 * where it is not defined. We define it to be zero in this case as well.
 */
template <size_t N, typename DataType>
DataType smoothstep_deriv(const double lower_edge, const double upper_edge,
                          const DataType& arg) {
  if (lower_edge == upper_edge) {
    return make_with_value<DataType>(arg, 0.);
  }
  ASSERT(lower_edge < upper_edge,
         "Requires lower_edge < upper_edge, but lower_edge="
             << lower_edge << " and upper_edge=" << upper_edge);
  using std::clamp;
  const DataType x = clamp(
      static_cast<DataType>((arg - lower_edge) / (upper_edge - lower_edge)), 0.,
      1.);
  constexpr auto coeff = (2 * N + 1) * binomial(2 * N, N);
  return coeff * pow<N>(x - square(x)) / (upper_edge - lower_edge);
}

/// \ingroup UtilitiesGroup
/// \brief Defines the inverse square-root (\f$1/\sqrt{x}\f$) for arithmetic
/// and complex types
template <typename T, Requires<std::is_arithmetic<T>::value or
                               tt::is_a_v<std::complex, T>> = nullptr>
auto invsqrt(const T& arg) {
  return static_cast<T>(1.0) / sqrt(arg);
}

/// \ingroup UtilitiesGroup
/// \brief Defines the inverse cube-root (\f$1/\sqrt[3]{x}\f$) for arithmetic
/// types
template <typename T, Requires<std::is_arithmetic<T>::value> = nullptr>
auto invcbrt(const T& arg) {
  return static_cast<T>(1.0) / cbrt(arg);
}

namespace sgn_detail {
template <typename T>
constexpr T sgn(const T& val, std::true_type /*is_signed*/) {
  return static_cast<T>(static_cast<T>(0) < val) -
         static_cast<T>(val < static_cast<T>(0));
}

template <typename T>
constexpr T sgn(const T& val, std::false_type /*is_signed*/) {
  return static_cast<T>(static_cast<T>(0) < val);
}
}  // namespace sgn_detail

/// \ingroup UtilitiesGroup
/// \brief Compute the sign function of `val` defined as `1` if `val > 0`, `0`
/// if `val == 0`, and `-1` if `val < 0`.
template <typename T>
constexpr T sgn(const T& val) {
  return sgn_detail::sgn(val, std::is_signed<T>{});
}

/// \ingroup UtilitiesGroup
/// \brief Raises a double to the integer power n.
inline double integer_pow(const double x, const int e) {
  ASSERT(e >= 0, "Negative powers are not implemented");
  int ecount = e;
  int bitcount = 1;
  while (ecount >>= 1) {
    ++bitcount;
  }
  double result = 1.;
  while (bitcount) {
    result *= result;
    if ((e >> --bitcount) & 0x1) {
      result *= x;
    }
  }
  return result;
}
