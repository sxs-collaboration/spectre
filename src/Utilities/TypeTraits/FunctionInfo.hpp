// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TMPL.hpp"

namespace tt {
/// \cond
namespace detail {
template <typename T>
struct function_info_impl;

template <typename Ret, typename... Args>
struct function_info_impl<Ret(Args...)> {
  using return_type = Ret;
  using argument_types = tmpl::list<Args...>;
  using class_type = void;
};

template <typename Ret, typename... Args>
struct function_info_impl<Ret(Args...) noexcept> {
  using return_type = Ret;
  using argument_types = tmpl::list<Args...>;
  using class_type = void;
};

#define FUNCTION_INFO_IMPL_FUNCTION_PTR(MODIFIERS, NOEXCEPT_STATUS)        \
  template <typename Ret,                                                  \
            typename... Args> /* NOLINTNEXTLINE(misc-macro-parentheses) */ \
  struct function_info_impl<Ret (*MODIFIERS)(Args...) NOEXCEPT_STATUS> {   \
    using return_type = Ret;                                               \
    using argument_types = tmpl::list<Args...>;                            \
    using class_type = void;                                               \
  }

FUNCTION_INFO_IMPL_FUNCTION_PTR(, );
FUNCTION_INFO_IMPL_FUNCTION_PTR(, noexcept);
FUNCTION_INFO_IMPL_FUNCTION_PTR(const, );
FUNCTION_INFO_IMPL_FUNCTION_PTR(const, noexcept);
FUNCTION_INFO_IMPL_FUNCTION_PTR(volatile, );
FUNCTION_INFO_IMPL_FUNCTION_PTR(volatile, noexcept);
FUNCTION_INFO_IMPL_FUNCTION_PTR(const volatile, );
FUNCTION_INFO_IMPL_FUNCTION_PTR(const volatile, noexcept);
#undef FUNCTION_INFO_IMPL_FUNCTION_PTR

#define FUNCTION_INFO_IMPL_CLASS(MODIFIERS)                                \
  template <typename Ret, typename Class,                                  \
            typename... Args> /* NOLINTNEXTLINE(misc-macro-parentheses) */ \
  struct function_info_impl<Ret (Class::*)(Args...) MODIFIERS> {           \
    using return_type = Ret;                                               \
    using argument_types = tmpl::list<Args...>;                            \
    using class_type = Class;                                              \
  }

FUNCTION_INFO_IMPL_CLASS();
FUNCTION_INFO_IMPL_CLASS(const);
FUNCTION_INFO_IMPL_CLASS(noexcept);
FUNCTION_INFO_IMPL_CLASS(volatile);
FUNCTION_INFO_IMPL_CLASS(const noexcept);
FUNCTION_INFO_IMPL_CLASS(const volatile);
FUNCTION_INFO_IMPL_CLASS(const volatile noexcept);
FUNCTION_INFO_IMPL_CLASS(volatile noexcept);
#undef FUNCTION_INFO_IMPL_CLASS
}  // namespace detail
/// \endcond

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
using function_info = detail::function_info_impl<F>;
}  // namespace tt
