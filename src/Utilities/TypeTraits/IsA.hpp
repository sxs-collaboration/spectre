// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

namespace tt {
// @{
/// \ingroup TypeTraitsGroup
/// \brief Check if type `T` is a template specialization of `U`
///
/// \requires `U` is a class template
/// \effects If `T` is a template specialization of `U`, then inherits from
/// std::true_type, otherwise inherits from std::false_type
///
/// \usage
/// For any type `T` and class template `U`
/// \code
/// using result = tt::is_a<U, T>;
/// \endcode
/// \metareturns
/// std::bool_constant
///
/// \semantics
/// If the type `T` is a template specialization of the type `U`, then
/// \code
/// typename result::type = std::true_type;
/// \endcode
/// otherwise
/// \code
/// typename result::type = std::false_type;
/// \endcode
///
/// \example
/// \snippet Test_IsA.cpp is_a_example
/// \see is_std_array
/// \tparam T type to check
/// \tparam U the type that T might be a template specialization of
template <template <typename...> class U, typename T>
struct is_a : std::false_type {};

/// \cond HIDDEN_SYMBOLS
template <template <typename...> class U, typename... Args>
struct is_a<U, U<Args...>> : std::true_type {};
/// \endcond

/// \see is_a
template <template <typename...> class U, typename... Args>
constexpr bool is_a_v = is_a<U, Args...>::value;

/// \see is_a
template <template <typename...> class U, typename... Args>
using is_a_t = typename is_a<U, Args...>::type;
// @}

namespace detail {
template <template <typename> class U>
struct is_a_wrapper;

template <typename U, typename T>
struct wrapped_is_a;

template <template <typename> class U, typename T>
struct wrapped_is_a<is_a_wrapper<U>, T> : tt::is_a<U, T> {};
}  // namespace detail

template <template <typename> class U, typename T>
using is_a_lambda = detail::wrapped_is_a<detail::is_a_wrapper<U>, T>;
}  // namespace tt
