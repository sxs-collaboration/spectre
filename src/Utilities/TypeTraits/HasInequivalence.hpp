// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

namespace tt {
// @{
/// \ingroup TypeTraitsGroup
/// \brief Check if type `T` has operator!= defined.
///
/// \details
/// Inherits from std::true_type if the type `T` has operator!= defined,
/// otherwise inherits from std::false_type
///
/// \usage
/// For any type `T`,
/// \code
/// using result = tt::has_inequivalence<T>;
/// \endcode
///
/// \metareturns
/// std::bool_constant
///
/// \semantics
/// If the type `T` has operator!= defined, then
/// \code
/// typename result::type = std::true_type;
/// \endcode
/// otherwise
/// \code
/// typename result::type = std::false_type;
/// \endcode
///
/// \example
/// \snippet Test_HasInequivalence.cpp has_inequivalence_example
/// \see has_equivalence
/// \tparam T the type we want to know if it has operator!=
template <typename T, typename U = void>
struct has_inequivalence : std::false_type {};

/// \cond HIDDEN_SYMBOLS
template <typename T>
struct has_inequivalence<
    T, std::void_t<decltype(std::declval<T>() != std::declval<T>())>>
    : std::true_type {};
/// \endcond

/// \see has_inequivalence
template <typename T>
constexpr bool has_inequivalence_v = has_inequivalence<T>::value;

/// \see has_inequivalence
template <typename T>
using has_inequivalence_t = typename has_inequivalence<T>::type;
// @}
}  // namespace tt
