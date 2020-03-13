// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines type traits, some of which are future STL type_traits header
///
/// Features present in C++17 are in the "cpp17" namespace.

#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <forward_list>
#include <functional>  // for reference_wrapper
#include <future>
#include <list>
#include <map>
#include <memory>
#include <ostream>
#include <queue>
#include <set>
#include <stack>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utilities/NoSuchType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StlStreamDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup TypeTraitsGroup
/// C++ STL code present in C++17
namespace cpp17 {

/*!
 * \ingroup UtilitiesGroup
 * \brief Mark a return type as being "void". In C++17 void is a regular type
 * under certain circumstances, so this can be replaced by `void` then.
 *
 * The proposal is available
 * [here](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0146r1.html)
 */
struct void_type {};

/// \ingroup TypeTraitsGroup
/// \brief A compile-time boolean
///
/// \usage
/// For any bool `B`
/// \code
/// using result = cpp17::bool_constant<B>;
/// \endcode
///
/// \see std::bool_constant std::integral_constant std::true_type
/// std::false_type
template <bool B>
using bool_constant = std::integral_constant<bool, B>;

// @{
/// \ingroup TypeTraitsGroup
/// \brief A logical AND on the template parameters
///
/// \details
/// Given a list of ::bool_constant template parameters computes their
/// logical AND. If the result is `true` then derive off of std::true_type,
/// otherwise derive from std::false_type. See the documentation for
/// std::conjunction for more details.
///
/// \usage
/// For any set of types `Ti` that are ::bool_constant like
/// \code
/// using result = cpp17::conjunction<T0, T1, T2>;
/// \endcode
/// \pre For all types `Ti`, `Ti::value` is a `bool`
///
/// \metareturns
/// ::bool_constant
///
/// \semantics
/// If `T::value != false` for all `Ti`, then
/// \code
/// using result = cpp17::bool_constant<true>;
/// \endcode
/// otherwise
/// \code
/// using result = cpp17::bool_constant<false>;
/// \endcode
///
/// \example
/// \snippet Utilities/Test_TypeTraits.cpp conjunction_example
/// \see std::conjunction, disjunction, std::disjunction
template <class...>
struct conjunction : std::true_type {};
/// \cond HIDDEN_SYMBOLS
template <class B1>
struct conjunction<B1> : B1 {};
template <class B1, class... Bn>
struct conjunction<B1, Bn...>
    : std::conditional_t<static_cast<bool>(B1::value), conjunction<Bn...>, B1> {
};
/// \endcond
/// \see std::conjunction
template <class... B>
constexpr bool conjunction_v = conjunction<B...>::value;
// @}

// @{
/// \ingroup TypeTraitsGroup
/// \brief A logical OR on the template parameters
///
/// \details
/// Given a list of ::bool_constant template parameters computes their
/// logical OR. If the result is `true` then derive off of std::true_type,
/// otherwise derive from std::false_type. See the documentation for
/// std::conjunction for more details.
///
/// \usage
/// For any set of types `Ti` that are ::bool_constant like
/// \code
/// using result = cpp17::disjunction<T0, T1, T2>;
/// \endcode
/// \pre For all types `Ti`, `Ti::value` is a `bool`
///
/// \metareturns
/// ::bool_constant
///
/// \semantics
/// If `T::value == true` for any `Ti`, then
/// \code
/// using result = cpp17::bool_constant<true>;
/// \endcode
/// otherwise
/// \code
/// using result = cpp17::bool_constant<false>;
/// \endcode
///
/// \example
/// \snippet Utilities/Test_TypeTraits.cpp disjunction_example
/// \see std::disjunction, conjunction, std::conjunction
template <class...>
struct disjunction : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <class B1>
struct disjunction<B1> : B1 {};
template <class B1, class... Bn>
struct disjunction<B1, Bn...>
    : std::conditional_t<static_cast<bool>(B1::value), B1, disjunction<Bn...>> {
};
/// \endcond
/// \see std::disjunction
template <class... B>
constexpr bool disjunction_v = disjunction<B...>::value;
// @}

/// \ingroup TypeTraitsGroup
/// \brief Negate a ::bool_constant
///
/// \details
/// Given a ::bool_constant returns the logical NOT of it.
///
/// \usage
/// For a ::bool_constant `B`
/// \code
/// using result = cpp17::negate<B>;
/// \endcode
///
/// \metareturns
/// ::bool_constant
///
/// \semantics
/// If `B::value == true` then
/// \code
/// using result = cpp17::bool_constant<false>;
/// \endcode
///
/// \example
/// \snippet Utilities/Test_TypeTraits.cpp negation_example
/// \see std::negation
/// \tparam B the ::bool_constant to negate
template <class B>
struct negation : bool_constant<!B::value> {};

/// \ingroup TypeTraitsGroup
/// \brief Given a set of types, returns `void`
///
/// \details
/// Given a list of types, returns `void`. This is very useful for controlling
/// name lookup resolution via partial template specialization.
///
/// \usage
/// For any set of types `Ti`,
/// \code
/// using result = cpp17::void_t<T0, T1, T2, T3>;
/// \endcode
///
/// \metareturns
/// void
///
/// \semantics
/// For any set of types `Ti`,
/// \code
/// using result = void;
/// \endcode
///
/// \example
/// \snippet Utilities/Test_TypeTraits.cpp void_t_example
/// \see std::void_t
/// \tparam Ts the set of types
template <typename... Ts>
using void_t = void;

/*!
 * \ingroup TypeTraitsGroup
 * \brief Variable template for is_same
 */
template <typename T, typename U>
constexpr bool is_same_v = std::is_same<T, U>::value;

/// \ingroup TypeTraitsGroup
template <typename T>
constexpr bool is_lvalue_reference_v = std::is_lvalue_reference<T>::value;

/// \ingroup TypeTraitsGroup
template <typename T>
constexpr bool is_rvalue_reference_v = std::is_rvalue_reference<T>::value;

/// \ingroup TypeTraitsGroup
template <typename T>
constexpr bool is_reference_v = std::is_reference<T>::value;

/// \ingroup TypeTraitsGroup
template <class T, class... Args>
using is_constructible_t = typename std::is_constructible<T, Args...>::type;

/// \ingroup TypeTraitsGroup
template <class T, class... Args>
constexpr bool is_constructible_v = std::is_constructible<T, Args...>::value;

/// \ingroup TypeTraitsGroup
template <class T, class... Args>
constexpr bool is_trivially_constructible_v =
    std::is_trivially_constructible<T, Args...>::value;

/// \ingroup TypeTraitsGroup
template <class T, class... Args>
constexpr bool is_nothrow_constructible_v =
    std::is_nothrow_constructible<T, Args...>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_default_constructible_v =
    std::is_default_constructible<T>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_trivially_default_constructible_v =
    std::is_trivially_default_constructible<T>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_nothrow_default_constructible_v =
    std::is_nothrow_default_constructible<T>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_copy_constructible_v = std::is_copy_constructible<T>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_trivially_copy_constructible_v =
    std::is_trivially_copy_constructible<T>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_nothrow_copy_constructible_v =
    std::is_nothrow_copy_constructible<T>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_move_constructible_v = std::is_move_constructible<T>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_trivially_move_constructible_v =
    std::is_trivially_move_constructible<T>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_nothrow_move_constructible_v =
    std::is_nothrow_move_constructible<T>::value;

/// \ingroup TypeTraitsGroup
template <class T, class U>
constexpr bool is_assignable_v = std::is_assignable<T, U>::value;

/// \ingroup TypeTraitsGroup
template <class T, class U>
constexpr bool is_trivially_assignable_v =
    std::is_trivially_assignable<T, U>::value;

/// \ingroup TypeTraitsGroup
template <class T, class U>
constexpr bool is_nothrow_assignable_v =
    std::is_nothrow_assignable<T, U>::value;

/// \ingroup TypeTraitsGroup
template <class From, class To>
constexpr bool is_convertible_v = std::is_convertible<From, To>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_copy_assignable_v = std::is_copy_assignable<T>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_trivially_copy_assignable_v =
    std::is_trivially_copy_assignable<T>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_nothrow_copy_assignable_v =
    std::is_nothrow_copy_assignable<T>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_move_assignable_v = std::is_move_assignable<T>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_trivially_move_assignable_v =
    std::is_trivially_move_assignable<T>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_nothrow_move_assignable_v =
    std::is_nothrow_move_assignable<T>::value;

/// \ingroup TypeTraitsGroup
template <class Base, class Derived>
constexpr bool is_base_of_v = std::is_base_of<Base, Derived>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_unsigned_v = std::is_unsigned<T>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_arithmetic_v = std::is_arithmetic<T>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_floating_point_v = std::is_floating_point<T>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_integral_v = std::is_integral<T>::value;

/// \ingroup TypeTraitsGroup
template <class T>
constexpr bool is_fundamental_v = std::is_fundamental<T>::value;

}  // namespace cpp17

/// \ingroup TypeTraitsGroup
/// C++ STL code present in C++20
namespace cpp20 {
/// \ingroup TypeTraitsGroup
template <class T>
struct remove_cvref {
  // clang-tidy use using instead of typedef
  typedef std::remove_cv_t<std::remove_reference_t<T>> type;  // NOLINT
};

/// \ingroup TypeTraitsGroup
template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;
}  // namespace cpp20

/// \ingroup TypeTraitsGroup
/// A collection of useful type traits
namespace tt {
// @{
/*!
 * \ingroup TypeTraitsGroup
 * \brief Check if `T` is copy constructible
 *
 * The STL `std::is_copy_constructible` does not work as expected with some
 * types, such as `std::unordered_map`. This is because
 * `std::is_copy_constructible` only checks that the copy construction call is
 * well-formed, not that it could actually be done in practice. To get around
 * this for containers we check that `T::value_type` is also copy constructible.
 */
template <typename T, typename = void>
struct can_be_copy_constructed : std::is_copy_constructible<T> {};

/// \cond
template <typename T>
struct can_be_copy_constructed<T, cpp17::void_t<typename T::value_type>>
    : cpp17::bool_constant<
          cpp17::is_copy_constructible_v<T> and
          cpp17::is_copy_constructible_v<typename T::value_type>> {};
/// \endcond

template <typename T>
constexpr bool can_be_copy_constructed_v = can_be_copy_constructed<T>::value;
// @}

// @{
/// \ingroup TypeTraitsGroup
/// \brief Check if type T is a std::array
///
/// \details
/// Given a type `T` derives from std::true_type if `T` is a std::array and from
/// std::false_type if `T` is not a std::array.
///
/// \usage
/// For any type `T`
/// \code
/// using result = tt::is_std_array<T>;
/// \endcode
///
/// \metareturns
/// cpp17::bool_constant
///
/// \semantics
/// If `T` is a std::array then
/// \code
/// typename result::type = cpp17::bool_constant<true>;
/// \endcode
/// otherwise
/// \code
/// typename result::type = cpp17::bool_constant<false>;
/// \endcode
///
/// \example
/// \snippet Utilities/Test_TypeTraits.cpp is_std_array_example
/// \see is_a is_std_array_of_size
/// \tparam T the type to check
template <typename T>
struct is_std_array : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename T, size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};
/// \endcond
/// \see is_std_array
template <typename T>
constexpr bool is_std_array_v = is_std_array<T>::value;

/// \see is_std_array
template <typename T>
using is_std_array_t = typename is_std_array<T>::type;
// @}

// @{
/// \ingroup TypeTraitsGroup
/// \brief Check if type T is a std::array of a given size
///
/// \details
/// Given a size_t `N` and type `T` derives from std::true_type if `T`
/// is a std::array of size `N` and from std::false_type otherwise.
///
/// \usage
/// For any type `T`
/// \code
/// using result = tt::is_std_array_of_size<N, T>;
/// \endcode
///
/// \metareturns
/// cpp17::bool_constant
///
/// \semantics
/// If `T` is a std::array of size `N` then
/// \code
/// typename result::type = cpp17::bool_constant<true>;
/// \endcode
/// otherwise
/// \code
/// typename result::type = cpp17::bool_constant<false>;
/// \endcode
///
/// \example
/// \snippet Utilities/Test_TypeTraits.cpp is_std_array_of_size_example
/// \see is_std_array
/// \tparam T the type to check
template <size_t N, typename T>
struct is_std_array_of_size : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <size_t N, typename T>
struct is_std_array_of_size<N, std::array<T, N>> : std::true_type {};
/// \endcond
/// \see is_std_array_of_size
template <size_t N, typename T>
constexpr bool is_std_array_of_size_v = is_std_array_of_size<N, T>::value;

/// \see is_std_array_of_size
template <size_t N, typename T>
using is_std_array_of_size_t = typename is_std_array_of_size<N, T>::type;
// @}

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
/// cpp17::bool_constant
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
/// \snippet Utilities/Test_TypeTraits.cpp is_a_example
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
/// cpp17::bool_constant
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
/// \snippet Utilities/Test_TypeTraits.cpp is_iterable_example
/// \see has_size
/// \tparam T the type to check
template <typename T, typename = cpp17::void_t<>>
struct is_iterable : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename T>
struct is_iterable<T, cpp17::void_t<decltype(std::declval<T>().begin(),
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

// @{
/// \ingroup TypeTraitsGroup
/// \brief Check if type T has <, <=, >, >=, ==, !=
///
/// \details
/// Given a type `T` inherits from std::true_type if `T` has operators `<`,
/// `<=`, `>`, `>=`, `==`, and `!=` defined, otherwise inherits from
/// std::false_type.
///
/// \usage
/// For any type `T`
/// \code
/// using result = tt::is_comparable<T>;
/// \endcode
///
/// \metareturns
/// cpp17::bool_constant
///
/// \semantics
/// If `T` has operators `<`, `<=`, `>`, `>=`, `==`, and `!=` defined, then
/// \code
/// typename result::type = std::true_type;
/// \endcode
/// otherwise
/// \code
/// typename result::type = std::false_type;
/// \endcode
///
/// \example
/// \snippet Utilities/Test_TypeTraits.cpp is_comparable_example
/// \see has_equivalence has_inequivalence
/// \tparam T the type to check
template <typename T, typename = cpp17::void_t<>>
struct is_comparable : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename T>
struct is_comparable<
    T, cpp17::void_t<decltype(std::declval<T>() < std::declval<T>()),
                     decltype(std::declval<T>() <= std::declval<T>()),
                     decltype(std::declval<T>() > std::declval<T>()),
                     decltype(std::declval<T>() >= std::declval<T>()),
                     decltype(std::declval<T>() == std::declval<T>()),
                     decltype(std::declval<T>() != std::declval<T>())>>
    : std::true_type {};
/// \endcond
/// \see is_comparable
template <typename T>
constexpr bool is_comparable_v = is_comparable<T>::value;

/// \see is_comparable
template <typename T>
using is_comparable_t = typename is_comparable<T>::type;
// @}

namespace TypeTraits_detail {
template <typename T, std::size_t N>
std::integral_constant<std::size_t, N> array_size_impl(
    const std::array<T, N>& /*array*/);
}  // namespace TypeTraits_detail

// @{
/// \ingroup TypeTraitsGroup
/// \brief Get the size of a std::array as a std::integral_constant
///
/// \details
/// Given a std::array, `Array`, returns a std::integral_constant that has the
/// size of the array as its value
///
/// \usage
/// For a std::array `T`
/// \code
/// using result = tt::array_size<T>;
/// \endcode
///
/// \metareturns
/// std::integral_constant<std::size_t>
///
/// \semantics
/// For a type `T`,
/// \code
/// using tt::array_size<std::array<T, N>> = std::integral_constant<std::size_t,
/// N>;
/// \endcode
///
/// \example
/// \snippet Utilities/Test_TypeTraits.cpp array_size_example
/// \tparam Array the whose size should be stored in value of array_size
template <typename Array>
using array_size =
    decltype(TypeTraits_detail::array_size_impl(std::declval<const Array&>()));
// @}

// @{
/// \ingroup TypeTraitsGroup
/// \brief Check if type `T` has operator== defined.
///
/// \details
/// Inherits from std::true_type if the type `T` has operator== defined,
/// otherwise inherits from std::false_type
///
/// \usage
/// For any type `T`,
/// \code
/// using result = tt::has_equivalence<T>;
/// \endcode
///
/// \metareturns
/// cpp17::bool_constant
///
/// \semantics
/// If the type `T` has operator== defined, then
/// \code
/// typename result::type = std::true_type;
/// \endcode
/// otherwise
/// \code
/// typename result::type = std::false_type;
/// \endcode
///
/// \example
/// \snippet Utilities/Test_TypeTraits.cpp has_equivalence_example
/// \see is_comparable has_inequivalence
/// \tparam T the type we want to know if it has operator==
template <typename T, typename = cpp17::void_t<>>
struct has_equivalence : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename T>
struct has_equivalence<
    T, cpp17::void_t<decltype(std::declval<T>() == std::declval<T>())>>
    : std::true_type {};
/// \endcond
/// \see has_equivalence
template <typename T>
constexpr bool has_equivalence_v = has_equivalence<T>::value;

/// \see has_equivalence
template <typename T>
using has_equivalence_t = typename has_equivalence<T>::type;
// @}

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
/// cpp17::bool_constant
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
/// \snippet Utilities/Test_TypeTraits.cpp has_equivalence_example
/// \see is_comparable has_equivalence
/// \tparam T the type we want to know if it has operator!=
template <typename T, typename U = void>
struct has_inequivalence : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename T>
struct has_inequivalence<
    T, cpp17::void_t<decltype(std::declval<T>() != std::declval<T>())>>
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
