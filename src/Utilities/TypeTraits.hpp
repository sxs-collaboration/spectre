// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines type traits, some of which are future STL type_traits header
///
/// Features present in C++17 are in the "cpp17" namespace.

#pragma once

#include <array>
#include <complex>
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

// @{
/// \ingroup TypeTraitsGroup
/// \brief Check if a type `T` is callable, i.e. `T(Args...)` is evaluable.
///
/// \details
/// Inherits from std::true_type if `TT` has the call operator, operator()
/// defined with arguments `TArgs...`, otherwise inherits from std::false_type.
///
/// \usage
/// For any type `TT` and types `TArgs_i`,
/// \code
/// using result = tt::is_callable<TT, TArgs_0, TArgs_1, TArgs_2>;
/// \endcode
///
/// \metareturns
/// cpp17::bool_constant
///
/// \semantics
/// If the type `TT` defines operator() with arguments `TArgs...`, then
/// \code
/// typename result::type = std::true_type;
/// \endcode
/// otherwise
/// \code
/// typename result::type = std::false_type;
/// \endcode
///
/// \example
/// \snippet Utilities/Test_TypeTraits.cpp is_callable_example
/// \see std::is_callable
/// \tparam TT the class to check
/// \tparam TArgs the args passed to operator()
template <typename TT, typename... TArgs>
class is_callable {
  // The reason we have private before public here is that we have static member
  // functions and since this is meant to be a super lightweight helper class
  // it's better to break convention than increase code size.
 private:
  /// \cond
  // We pass an int here to disambiguate the two possible templates and have the
  // compiler prefer the first one. If it cannot be used because there's no
  // call operator, then it uses the second one.
  template <typename T, typename... Args>
  static auto test_callable(int) noexcept
      -> decltype(std::declval<T>()(std::declval<Args>()...), std::true_type());

  template <typename, typename...>
  static auto test_callable(...) noexcept -> std::false_type;
  /// \endcond

 public:
  /// `true` if callable, `false` otherwise
  static constexpr bool value = decltype(test_callable<TT, TArgs...>(0))::value;
  /// `std::true_type` if callable, `std::false_type` otherwise
  using type = std::integral_constant<bool, value>;
};
/// \see is_callable
template <typename T, typename... Args>
constexpr bool is_callable_v = is_callable<T, Args...>::value;

/// \see is_callable
template <typename T, typename... Args>
using is_callable_t = typename is_callable<T, Args...>::type;
// @}

/*!
 * \ingroup TypeTraitsGroup
 * \brief Generate a type trait to check if a class has a member function that
 * can be invoked with arguments of type `TArgs...`
 *
 * The usage of the type trait is identical to the usage of the
 * `tt::is_callable` type trait. The name of the type trait is
 * `is_METHOD_NAME_callable` and is not placed in
 * the `tt` namespace. To avoid collisions it is highly recommended that type
 * traits generated with this macro are generated into `_detail` namespaces.
 * This will reduce redefinition compilation errors. Note that a variable
 * template `is_METHOD_NAME_callable_v`, and a `is_METHOD_NAME_callable_t` that
 * is either `std::true_type` or `std::false_type` is also generated.
 *
 * \example
 * \snippet Utilities/Test_TypeTraits.cpp CREATE_IS_CALLABLE_EXAMPLE
 *
 * \see tt::is_callable
 */
#define CREATE_IS_CALLABLE(METHOD_NAME)                                     \
  template <typename TT, typename... TArgs>                                 \
  class is_##METHOD_NAME##_callable {                                       \
   private:                                                                 \
    template <typename T, typename... Args>                                 \
    static auto test_callable(int) noexcept                                 \
        -> decltype(std::declval<T>().METHOD_NAME(std::declval<Args>()...), \
                    std::true_type());                                      \
    template <typename, typename...>                                        \
    static auto test_callable(...) noexcept -> std::false_type;             \
                                                                            \
   public:                                                                  \
    static constexpr bool value =                                           \
        decltype(test_callable<TT, TArgs...>(0))::value;                    \
    using type = std::integral_constant<bool, value>;                       \
  };                                                                        \
  template <typename T, typename... Args>                                   \
  static constexpr const bool is_##METHOD_NAME##_callable_v =               \
      is_##METHOD_NAME##_callable<T, Args...>::value;                       \
  template <typename T, typename... Args>                                   \
  using is_##METHOD_NAME##_callable_t =                                     \
      typename is_##METHOD_NAME##_callable<T, Args...>::type;

// @{
/// \ingroup TypeTraitsGroup
/// \brief Check if std::hash and std::equal_to are defined for type T
///
/// \details
/// Inherits from std::true_type if std::hash and std::equal_to are defined for
/// the type `T`, otherwise inherits from std::false_type.
///
/// \usage
/// For any type `T`
/// \code
/// using result = tt::is_hashable<T>;
/// \endcode
///
/// \metareturns
/// cpp17::bool_constant
///
/// \semantics
/// If std::hash and std::equal_to are defined for the type `T`, then
/// \code
/// typename result::type = std::true_type;
/// \endcode
/// otherwise
/// \code
/// typename result::type = std::false_type;
/// \endcode
///
/// \example
/// \snippet Utilities/Test_TypeTraits.cpp is_hashable_example
/// \see std::hash std::equal_to
/// \tparam T type to check
template <typename T, typename = cpp17::void_t<>>
struct is_hashable : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename T>
struct is_hashable<T,
                   cpp17::void_t<decltype(std::hash<T>{}, std::equal_to<T>{})>>
    : std::true_type {};
/// \endcond
/// \see is_hashable
template <typename T>
constexpr bool is_hashable_v = is_hashable<T>::value;

/// \see is_hashable
template <typename T>
using is_hashable_t = typename is_hashable<T>::type;
// @}

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
/// \snippet Utilities/Test_TypeTraits.cpp is_maplike_example
/// \see std::map std::unordered_map is_a
/// \tparam T the type to check
template <typename T, typename = cpp17::void_t<>>
struct is_maplike : std::false_type {};
/// \cond HIDDEN_SYMBOLS
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

// @{
/// \ingroup TypeTraitsGroup
/// \brief Check if type `T` has operator<<(`S`, `T`) defined.
///
/// \details
/// Inherits from std::true_type if the type `T` has operator<<(`S`, `T`)
/// defined for a stream `S`, otherwise inherits from std::false_type
///
/// \usage
/// For any type `T` and stream type `S`,
/// \code
/// using result = tt::is_streamable<S, T>;
/// \endcode
///
/// \metareturns
/// cpp17::bool_constant
///
/// \semantics
/// If the type `T` has operator<<(`S`, `T`) defined for stream `S`, then
/// \code
/// typename result::type = std::true_type;
/// \endcode
/// otherwise
/// \code
/// typename result::type = std::false_type;
/// \endcode
///
/// \example
/// \snippet Utilities/Test_TypeTraits.cpp is_streamable_example
/// \see std::cout std::ifstream std::sstream std::ostream
/// \tparam S the stream type, e.g. std::stringstream or std::ostream
/// \tparam T the type we want to know if it has operator<<
template <typename S, typename T, typename = cpp17::void_t<>>
struct is_streamable : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename S, typename T>
struct is_streamable<
    S, T, cpp17::void_t<decltype(std::declval<std::add_lvalue_reference_t<S>>()
                                 << std::declval<T>()),
                        Requires<not std::is_same<S, T>::value>>>
    : std::true_type {};
/// \endcond
/// \see is_streamable
template <typename S, typename T>
constexpr bool is_streamable_v = is_streamable<S, T>::value;

/// \see is_streamable
template <typename S, typename T>
using is_streamable_t = typename is_streamable<S, T>::type;

// @}

// @{
/// \ingroup TypeTraitsGroup
/// \brief Check if type `T` is a std::string, or a C-style string
///
/// \details
/// Inherits from std::true_type if the type `T` has a is a std::string, or
/// a C-style string, otherwise inherits from std::false_type
///
/// \usage
/// For any type `T`,
/// \code
/// using result = tt::is_string_like<T>;
/// \endcode
///
/// \metareturns
/// cpp17::bool_constant
///
/// \semantics
/// If the type `T` is a std::string, or a C-style string, then
/// \code
/// typename result::type = std::true_type;
/// \endcode
/// otherwise
/// \code
/// typename result::type = std::false_type;
/// \endcode
///
/// \example
/// \snippet Utilities/Test_TypeTraits.cpp is_string_like_example
/// \see std::string std::is_same
/// \tparam T the type to check
template <typename T, typename = std::nullptr_t>
struct is_string_like : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename T>
struct is_string_like<
    T,
    Requires<std::is_same<std::decay_t<T>, std::string>::value or
             std::is_same<std::decay_t<std::remove_pointer_t<std::decay_t<T>>>,
                          char>::value>> : std::true_type {};
/// \endcond
/// \see is_string_like
template <typename T>
constexpr bool is_string_like_v = is_string_like<T>::value;

/// \see is_string_like
template <typename T>
using is_string_like_t = typename is_string_like<T>::type;

// @}

// @{
/// \ingroup TypeTraitsGroup
/// \brief Check if type `T` has a `get_clone()` member function
///
/// \details
/// Inherits from std::true_type if the type `T` has a `get_clone()` member
/// function, otherwise inherits from std::false_type
///
/// \usage
/// For any type `T`,
/// \code
/// using result = tt::has_get_clone<T>;
/// \endcode
///
/// \metareturns
/// cpp17::bool_constant
///
/// \semantics
/// If the type `T` has a `get_clone()` member function, then
/// \code
/// typename result::type = std::true_type;
/// \endcode
/// otherwise
/// \code
/// typename result::type = std::false_type;
/// \endcode
///
/// \example
/// \snippet Utilities/Test_TypeTraits.cpp has_get_clone_example
/// \see has_clone
/// \tparam T the type to check
template <typename T, typename = cpp17::void_t<>, typename = std::nullptr_t>
struct has_get_clone : std::false_type {};
/// \cond HIDDEN_SYMBOLS
// The ugliness with two void template parameters is because clang does not
// recognize the two decltype'd specializations as being different
template <typename T>
struct has_get_clone<
    T, cpp17::void_t<decltype(
           std::declval<std::remove_pointer_t<std::decay_t<T>>>().get_clone())>,
    Requires<not tt::is_a_v<std::unique_ptr, std::decay_t<T>> and
             not tt::is_a_v<std::shared_ptr, std::decay_t<T>>>>
    : std::true_type {};
template <typename T>
struct has_get_clone<T, cpp17::void_t<Requires<tt::is_a_v<std::unique_ptr, T>>,
                                      decltype(std::declval<T>()->get_clone())>,
                     Requires<tt::is_a_v<std::unique_ptr, std::decay_t<T>> or
                              tt::is_a_v<std::shared_ptr, std::decay_t<T>>>>
    : std::true_type {};
/// \endcond
/// \see has_get_clone
template <typename T>
constexpr bool has_get_clone_v = has_get_clone<T>::value;

/// \see has_get_clone
template <typename T>
using has_get_clone_t = typename has_get_clone<T>::type;
// @}

// @{
/// \ingroup TypeTraitsGroup
/// \brief Check if type `T` has a `clone()` member function
///
/// \details
/// Inherits from std::true_type if the type `T` has a `clone()` member
/// function, otherwise inherits from std::false_type
///
/// \usage
/// For any type `T`,
/// \code
/// using result = tt::has_clone<T>;
/// \endcode
///
/// \metareturns
/// cpp17::bool_constant
///
/// \semantics
/// If the type `T` has a `clone()` member function, then
/// \code
/// typename result::type = std::true_type;
/// \endcode
/// otherwise
/// \code
/// typename result::type = std::false_type;
/// \endcode
///
/// \example
/// \snippet Utilities/Test_TypeTraits.cpp has_clone_example
/// \see has_get_clone
/// \tparam T the type to check
template <typename T, typename = cpp17::void_t<>, typename = std::nullptr_t>
struct has_clone : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename T>
struct has_clone<T, cpp17::void_t<decltype(std::declval<T>().clone())>,
                 Requires<not tt::is_a_v<std::unique_ptr, std::decay_t<T>> and
                          not tt::is_a_v<std::shared_ptr, std::decay_t<T>>>>
    : std::true_type {};
template <typename T>
struct has_clone<T, cpp17::void_t<decltype(std::declval<T>()->clone())>,
                 Requires<tt::is_a_v<std::unique_ptr, std::decay_t<T>> or
                          tt::is_a_v<std::shared_ptr, std::decay_t<T>>>>
    : std::true_type {};
/// \endcond
/// \see has_clone
template <typename T>
constexpr bool has_clone_v = has_clone<T>::value;

/// \see has_clone
template <typename T>
using has_clone_t = typename has_clone<T>::type;
// @}

// @{
/// \ingroup TypeTraitsGroup
/// \brief Check if type `T` has a `size()` member function
///
/// \details
/// Inherits from std::true_type if the type `T` has a `size()` member
/// function, otherwise inherits from std::false_type
///
/// \usage
/// For any type `T`,
/// \code
/// using result = tt::has_size<T>;
/// \endcode
/// \metareturns
/// cpp17::bool_constant
///
/// \semantics
/// If the type `T` has a `size()` member function, then
/// \code
/// typename result::type = std::true_type;
/// \endcode
/// otherwise
/// \code
/// typename result::type = std::false_type;
/// \endcode
///
/// \example
/// \snippet Utilities/Test_TypeTraits.cpp has_size_example
/// \see is_iterable
/// \tparam T the type to check
template <typename T, typename = cpp17::void_t<>>
struct has_size : std::false_type {};
/// \cond HIDDEN_SYMBOLS
template <typename T>
struct has_size<T, cpp17::void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};
/// \endcond
/// \see has_size
template <typename T>
constexpr bool has_size_v = has_size<T>::value;

/// \see has_size
template <typename T>
using has_size_t = typename has_size<T>::type;
// @}

// @{
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
 * cpp17::bool_constant
 *
 * \example
 * \snippet Utilities/Test_TypeTraits.cpp is_integer_example
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
// @}

// @{
/*!
 * \ingroup TypeTraitsGroup
 * \brief Gets the underlying type if the type is a std::reference_wrapper,
 * otherwise returns the type itself
 *
 * \usage
 * For any type `I`,
 * \code
 * using result = tt::remove_reference_wrapper<I>;
 * \endcode
 * \metareturns
 * either `I::type` if `I` is a std::reference_wrapper, else returns I
 *
 * \example
 * \snippet Utilities/Test_TypeTraits.cpp remove_reference_wrapper_example
 * \see std::reference_wrapper
 */
template <typename T>
struct remove_reference_wrapper {
  using type = T;
};
/// \cond HIDDEN_SYMBOLS
template <typename T>
struct remove_reference_wrapper<std::reference_wrapper<T>> {
  using type = T;
};
template <typename T>
struct remove_reference_wrapper<const std::reference_wrapper<T>> {
  using type = T;
};
template <typename T>
struct remove_reference_wrapper<volatile std::reference_wrapper<T>> {
  using type = T;
};
template <typename T>
struct remove_reference_wrapper<const volatile std::reference_wrapper<T>> {
  using type = T;
};

/// \endcond

template <typename T>
using remove_reference_wrapper_t = typename remove_reference_wrapper<T>::type;
// @}

// @{
/*!
 * \ingroup TypeTraitsGroup
 * \brief Removes std::reference_wrapper, references, and cv qualifiers.
 *
 * \example
 * \snippet Utilities/Test_TypeTraits.cpp remove_cvref_wrap
 * \see std::reference_wrapper remove_reference_wrapper std::remove_cvref
 */
template <typename T>
struct remove_cvref_wrap {
  using type = cpp20::remove_cvref_t<tt::remove_reference_wrapper_t<T>>;
};

template <typename T>
using remove_cvref_wrap_t = typename remove_cvref_wrap<T>::type;
// @}

// @{
/// \ingroup TypeTraitsGroup
/// \brief Extracts the fundamental type for a container
///
/// \details  Designates a type alias `get_fundamental_type::type`
///  as `T` when `T` itself is an appropriate fundamental type, and the
/// contained type of a container which specifies a `value_type`.
///
/// `get_fundamental_type_t<T>` is provided as a type alias to
/// `type` from `get_fundamental_type<T>`
template <typename T, typename Enable = cpp17::void_t<>>
struct get_fundamental_type {
  using type = tmpl::conditional_t<cpp17::is_fundamental_v<T>, T, NoSuchType>;
};
/// \cond
template <typename T>
struct get_fundamental_type<T, cpp17::void_t<typename T::value_type>> {
  using type = typename get_fundamental_type<typename T::value_type>::type;
};
/// \endcond
template <typename T>
using get_fundamental_type_t = typename get_fundamental_type<T>::type;
// @}

// @{
/// \ingroup TypeTraitsGroup
/// \brief Determines if a type `T` is a `std::complex` of a fundamental type,
/// is a `std::true_type` if so, and otherwise is a `std::false_type`
///
/// \snippet Test_TypeTraits.cpp is_complex_of_fundamental
template <typename T, typename = cpp17::bool_constant<true>>
struct is_complex_of_fundamental : std::false_type {};
/// \cond
// this version will only pattern match if `T` is both complex and a fundamental
// type
template <typename T>
struct is_complex_of_fundamental<
    std::complex<T>, cpp17::bool_constant<cpp17::is_fundamental_v<T>>>
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
    is_complex_of_fundamental_v<T> or cpp17::is_fundamental_v<T>;

namespace tt_detail {
template <typename T>
struct function_info_impl;

template <typename Ret, typename... Args>
struct function_info_impl<Ret(Args...)> {
  using return_type = Ret;
  using argument_types = tmpl::list<Args...>;
  using class_type = void;
};

template <typename Ret, typename... Args>
struct function_info_impl<Ret (*)(Args...)> {
  using return_type = Ret;
  using argument_types = tmpl::list<Args...>;
  using class_type = void;
};

template <typename Ret, typename... Args>
struct function_info_impl<Ret (*const)(Args...)> {
  using return_type = Ret;
  using argument_types = tmpl::list<Args...>;
  using class_type = void;
};

template <typename Ret, typename... Args>
struct function_info_impl<Ret (*const volatile)(Args...)> {
  using return_type = Ret;
  using argument_types = tmpl::list<Args...>;
  using class_type = void;
};

template <typename Ret, typename... Args>
struct function_info_impl<Ret (*volatile)(Args...)> {
  using return_type = Ret;
  using argument_types = tmpl::list<Args...>;
  using class_type = void;
};

template <typename Ret, typename Class, typename... Args>
struct function_info_impl<Ret (Class::*)(Args...)> {
  using return_type = Ret;
  using argument_types = tmpl::list<Args...>;
  using class_type = Class;
};

template <typename Ret, typename Class, typename... Args>
struct function_info_impl<Ret (Class::*)(Args...) const> {
  using return_type = Ret;
  using argument_types = tmpl::list<Args...>;
  using class_type = Class;
};

template <typename Ret, typename Class, typename... Args>
struct function_info_impl<Ret (Class::*)(Args...) const volatile> {
  using return_type = Ret;
  using argument_types = tmpl::list<Args...>;
  using class_type = Class;
};

template <typename Ret, typename Class, typename... Args>
struct function_info_impl<Ret (Class::*)(Args...) volatile> {
  using return_type = Ret;
  using argument_types = tmpl::list<Args...>;
  using class_type = Class;
};
}  // namespace tt_detail

/*!
 * \ingroup TypeTraitsGroup
 * \brief Returns a struct that contains the return type, argument types, and
 * the class type if the `F` is a non-static member function
 *
 * The return class has member type aliases:
 * - `return_type` The return type of the function
 * - `argument_types` A `tmpl::list` of the arguments types of the function
 * - `class_type` The type of the class if the function is a non-static member
 * function, otherwise `void`
 *
 * \note For static member variables the class will be `void` because they are
 * effectively free functions.
 */
template <typename F>
using function_info = tt_detail::function_info_impl<F>;

/*!
 * \ingroup TypeTraitsGroup
 * \brief Given a type `T` and possibly a `size_t` evaluates to `T`. Useful for
 * turning a `std::index_sequence` into a pack expansion of types.
 *
 * \snippet Test_TypeTraits.cpp example_identity_t
 */
template <typename T, size_t = 0>
using identity_t = T;
}  // namespace tt
