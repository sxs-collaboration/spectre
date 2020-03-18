// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TMPL.hpp"

namespace tt {
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
}  // namespace detail

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
