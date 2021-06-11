// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

namespace tt {
/// @{
/*!
 * \ingroup TypeTraitsGroup
 * \brief Check if `I` is an integer type (non-bool, non-character), unlike
 * std::is_integral
 *
 * \details
 * Inherits from `std::true_type` if `I` is a `short`, `unsigned short`,
 * `int`, `unsigned int`, `long`, `unsigned long`, `long long`, or
 * `unsigned long long`, otherwise inherits from `std::false_type`.
 *
 * \usage
 * For any type `I`,
 * \code
 * using result = tt::is_integer<I>;
 * \endcode
 * \metareturns
 * std::bool_constant
 *
 * \example
 * \snippet Test_IsInteger.cpp is_integer_example
 * \see std::is_integral std::is_arithmetic std::is_floating_point
 */
template <typename I>
struct is_integer : std::false_type {};

/// \cond HIDDEN_SYMBOLS
template <>
struct is_integer<short> : std::true_type {};

template <>
struct is_integer<unsigned short> : std::true_type {};

template <>
struct is_integer<int> : std::true_type {};

template <>
struct is_integer<unsigned int> : std::true_type {};

template <>
struct is_integer<long> : std::true_type {};

template <>
struct is_integer<unsigned long> : std::true_type {};

template <>
struct is_integer<long long> : std::true_type {};

template <>
struct is_integer<unsigned long long> : std::true_type {};
/// \endcond

/// \see is_integer
template <typename T>
constexpr bool is_integer_v = is_integer<T>::value;
/// @}
}  // namespace tt
