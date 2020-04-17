// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <type_traits>

namespace tt {
// @{
/// \ingroup TypeTraitsGroup
/// \brief Determines if a type `T` is a `std::complex` of a fundamental type,
/// is a `std::true_type` if so, and otherwise is a `std::false_type`
///
/// \snippet Test_IsComplexOfFundamental.cpp is_complex_of_fundamental
template <typename T, typename = std::bool_constant<true>>
struct is_complex_of_fundamental : std::false_type {};

/// \cond
// this version will only pattern match if `T` is both complex and a fundamental
// type
template <typename T>
struct is_complex_of_fundamental<std::complex<T>,
                                 std::bool_constant<std::is_fundamental_v<T>>>
    : std::true_type {};
/// \endcond
// @}

template <typename T>
constexpr bool is_complex_of_fundamental_v =
    is_complex_of_fundamental<T>::value;

/// \ingroup TypeTraitsGroup
/// \brief Evaluates to `true` if type `T` is a `std::complex` of a fundamental
/// type or if `T` is a fundamental type.
template <typename T>
constexpr bool is_complex_or_fundamental_v =
    is_complex_of_fundamental_v<T> or std::is_fundamental_v<T>;
}  // namespace tt
