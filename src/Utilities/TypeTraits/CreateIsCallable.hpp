// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

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
 * This will reduce redefinition compilation errors.
 *
 * Note that variable templates with `_r` and `_t` suffixes that follow the
 * standard library's naming convention are also generated. To generate
 * corresponding `_v` metafunctions, call `CREATE_IS_CALLABLE` first and then
 * `CREATE_IS_CALLABLE_V` and/or `CREATE_IS_CALLABLE_R_V`.
 *
 * \example
 * \snippet Test_CreateIsCallable.cpp CREATE_IS_CALLABLE_EXAMPLE
 *
 * \see tt::is_callable
 */
#define CREATE_IS_CALLABLE(METHOD_NAME)                                        \
  struct AnyReturnType##METHOD_NAME {};                                        \
                                                                               \
  template <typename ReturnType, typename TT, typename... TArgs>               \
  class is_##METHOD_NAME##_callable_r {                                        \
   private:                                                                    \
    struct NotCallable {};                                                     \
    template <typename T, typename... Args>                                    \
    static auto test_callable(int) noexcept                                    \
        -> decltype(std::declval<T>().METHOD_NAME(std::declval<Args>()...));   \
    template <typename, typename...>                                           \
    static auto test_callable(...) noexcept -> NotCallable;                    \
                                                                               \
   public:                                                                     \
    static constexpr bool value =                                              \
        (std::is_same_v<ReturnType, AnyReturnType##METHOD_NAME> and            \
         not std::is_same_v<decltype(test_callable<TT, TArgs...>(0)),          \
                            NotCallable>) or                                   \
        std::is_same_v<decltype(test_callable<TT, TArgs...>(0)), ReturnType>;  \
    using type = std::integral_constant<bool, value>;                          \
  };                                                                           \
  template <typename ReturnType, typename T, typename... Args>                 \
  using is_##METHOD_NAME##_callable_r_t =                                      \
      typename is_##METHOD_NAME##_callable_r<ReturnType, T, Args...>::type;    \
  template <typename TT, typename... TArgs>                                    \
  using is_##METHOD_NAME##_callable =                                          \
      is_##METHOD_NAME##_callable_r<AnyReturnType##METHOD_NAME, TT, TArgs...>; \
  template <typename TT, typename... TArgs>                                    \
  using is_##METHOD_NAME##_callable_t =                                        \
      is_##METHOD_NAME##_callable_r_t<AnyReturnType##METHOD_NAME, TT,          \
                                      TArgs...>;

// Separate macros to avoid compiler warnings about unused variables
#define CREATE_IS_CALLABLE_R_V(METHOD_NAME)                     \
  template <typename ReturnType, typename T, typename... Args>  \
  static constexpr const bool is_##METHOD_NAME##_callable_r_v = \
      is_##METHOD_NAME##_callable_r<ReturnType, T, Args...>::value;
#define CREATE_IS_CALLABLE_V(METHOD_NAME)                     \
  template <typename T, typename... Args>                     \
  static constexpr const bool is_##METHOD_NAME##_callable_v = \
      is_##METHOD_NAME##_callable<T, Args...>::value;
// @}
