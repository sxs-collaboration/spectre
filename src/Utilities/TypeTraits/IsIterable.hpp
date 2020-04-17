// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

namespace tt {
// @{
/// \ingroup TypeTraitsGroup
/// \brief Check if type T has a begin() and end() function
///
/// \details
/// Given a type `T` inherits from std::true_type if `T` has member functions
/// `begin()` and `end()`, otherwise inherits from std::false_type
///
/// \usage
/// For any type `T`
/// \code
/// using result = tt::is_iterable<T>;
/// \endcode
///
/// \metareturns
/// std::bool_constant
///
/// \semantics
/// If `T` has member function `begin()` and `end()` then
/// \code
/// typename result::type = std::true_type;
/// \endcode
/// otherwise
/// \code
/// typename result::type = std::false_type;
/// \endcode
///
/// \example
/// \snippet Test_IsIterable.cpp is_iterable_example
/// \tparam T the type to check
template <typename T, typename = std::void_t<>>
struct is_iterable : std::false_type {};

/// \cond HIDDEN_SYMBOLS
template <typename T>
struct is_iterable<T, std::void_t<decltype(std::declval<T>().begin(),
                                           std::declval<T>().end())>>
    : std::true_type {};
/// \endcond

/// \see is_iterable
template <typename T>
constexpr bool is_iterable_v = is_iterable<T>::value;

/// \see is_iterable
template <typename T>
using is_iterable_t = typename is_iterable<T>::type;
// @}
}  // namespace tt
