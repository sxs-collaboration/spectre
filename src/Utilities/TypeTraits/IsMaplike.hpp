// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/IsIterable.hpp"

namespace tt {
// @{
/// \ingroup TypeTraitsGroup
/// \brief Check if type `T` is like a std::map or std::unordored_map
///
/// \details
/// Inherits from std::true_type if the type `T` has a type alias `key_type`,
/// type alias `mapped_type`, and `operator[](const typename T::key_type&)`
/// defined, otherwise inherits from std::false_type
///
/// \usage
/// For any type `T`,
/// \code
/// using result = tt::is_maplike<T>;
/// \endcode
///
/// \metareturns
/// cpp17::bool_constant
///
/// \semantics
/// If the type `T` has a type alias `key_type`,
/// type alias `mapped_type`, and `operator[](const typename T::key_type&)`
/// defined, then
/// \code
/// typename result::type = std::true_type;
/// \endcode
/// otherwise
/// \code
/// typename result::type = std::false_type;
/// \endcode
///
/// \example
/// \snippet Test_IsMaplike.cpp is_maplike_example
/// \see std::map std::unordered_map is_a
/// \tparam T the type to check
template <typename T, typename = cpp17::void_t<>>
struct is_maplike : std::false_type {};

/// \cond
template <typename T>
struct is_maplike<T,
                  cpp17::void_t<typename T::key_type, typename T::mapped_type,
                                decltype(std::declval<T&>()[std::declval<
                                    const typename T::key_type&>()]),
                                Requires<tt::is_iterable_v<T>>>>
    : std::true_type {};
/// \endcond

/// \see is_maplike
template <typename T>
constexpr bool is_maplike_v = is_maplike<T>::value;

/// \see is_maplike
template <typename T>
using is_maplike_t = typename is_maplike<T>::type;
// @}
}  // namespace tt
