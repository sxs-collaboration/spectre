// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

namespace tt {
/// @{
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
/// std::bool_constant
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
/// \snippet Test_IsCallable.cpp is_callable_example
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
  static auto test_callable(int)
      -> decltype(std::declval<T>()(std::declval<Args>()...), std::true_type());

  template <typename, typename...>
  static auto test_callable(...) -> std::false_type;
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
/// @}
}  // namespace tt
