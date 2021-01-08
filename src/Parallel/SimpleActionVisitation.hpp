// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/variant/get.hpp>
#include <boost/variant/variant.hpp>
#include <exception>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"  // IWYU pragma: keep
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Parallel {
namespace Algorithm_detail {
template <typename TT, typename TParallelComponent, typename... TArgs>
class is_apply_callable {
 private:
  template <typename T, typename ParallelComponent, typename... Args>
  static auto test_callable(int) noexcept
      -> decltype(std::declval<T>().template apply<ParallelComponent>(
                      std::declval<Args>()...),
                  std::true_type());
  template <typename, typename...>
  static auto test_callable(...) noexcept -> std::false_type;

 public:
  static constexpr bool value =
      decltype(test_callable<TT, TParallelComponent, TArgs...>(0))::value;
  using type = std::integral_constant<bool, value>;
};

template <typename T, typename ParallelComponent, typename... Args>
static constexpr const bool is_apply_callable_v =
    is_apply_callable<T, ParallelComponent, Args...>::value;

template <typename T, typename ParallelComponent, typename... Args>
using is_apply_callable_t =
    typename is_apply_callable<T, ParallelComponent, Args...>::type;

template <typename Invokable, typename ParallelComponent, typename ThisVariant,
          typename... Variants, typename... Args,
          Requires<is_apply_callable_v<Invokable, ParallelComponent,
                                       std::add_lvalue_reference_t<ThisVariant>,
                                       Args&&...>> = nullptr>
void simple_action_visitor_helper(boost::variant<Variants...>& box,
                                  const gsl::not_null<int*> iter,
                                  const gsl::not_null<bool*> already_visited,
                                  Args&&... args) {
  if (box.which() == *iter and not*already_visited) {
    try {
      static_assert(
          std::is_same<void,
                       decltype(Invokable::template apply<ParallelComponent>(
                           std::declval<ThisVariant&>(),
                           std::declval<Args>()...))>::value,
          "A simple action must return void.");
      Invokable::template apply<ParallelComponent>(boost::get<ThisVariant>(box),
                                                   std::forward<Args>(args)...);
    } catch (std::exception& e) {
      ERROR("Fatal error: Failed to call simple Action '"
            << pretty_type::get_name<Invokable>() << "' on iteration '" << iter
            << "' with DataBox type '" << pretty_type::get_name<ThisVariant>()
            << "'\nThe exception is: '" << e.what() << "'\n");
    }
    *already_visited = true;
  }
  (*iter)++;
}

template <typename Invokable, typename ParallelComponent, typename ThisVariant,
          typename... Variants, typename... Args,
          Requires<not is_apply_callable_v<
              Invokable, ParallelComponent,
              std::add_lvalue_reference_t<ThisVariant>, Args&&...>> = nullptr>
void simple_action_visitor_helper(boost::variant<Variants...>& box,
                                  const gsl::not_null<int*> iter,
                                  const gsl::not_null<bool*> already_visited,
                                  Args&&... /*args*/) {
  if (box.which() == *iter and not*already_visited) {
    std::string args_list{};
    const auto helper = [&args_list](auto arg_v) {
      args_list +=
          pretty_type::get_name<typename decltype(arg_v)::type>() + ", ";
    };
    EXPAND_PACK_LEFT_TO_RIGHT(helper(tmpl::type_<Args>{}));
    if (sizeof...(Args) > 0) {
      args_list.pop_back();
      args_list.pop_back();
    }
    ERROR("\nCannot call apply function of '"
          << pretty_type::get_name<Invokable>() << "' with DataBox type '"
          << pretty_type::get_name<ThisVariant>() << "' and arguments '"
          << args_list << "'.\n"
          << "If the argument types to the apply function match, then it is "
             "possible that the apply function has non-deducible template "
             "parameters. This could occur from removing an apply function "
             "argument and forgetting to remove its associated template "
             "parameters.\n");
  }
  (*iter)++;
}

/*!
 * \brief Calls an `Invokable`'s `apply` static member function with the current
 * type in the `boost::variant`.
 *
 * The simple action can only be invoked if it accesses members of the DataBox
 * that are guaranteed to be present at the time of invocation. This can lead to
 * race conditions since simple actions can be invoked at any time during
 * execution at any part of any phase. As a concrete example of such a race
 * condition, consider a parallel component with iterable actions A, B, and C
 * where B adds something to the DataBox (call it G) and C removes it. If a
 * simple action S that uses G is called on the parallel component the call will
 * succeed if S is invoked between B and C, and fail otherwise.
 */
template <typename Invokable, typename ParallelComponent, typename... Variants,
          typename... Args>
void simple_action_visitor(boost::variant<Variants...>& box, Args&&... args) {
  // iter is the current element of the variant in the "for loop"
  int iter = 0;
  // already_visited ensures that only one visitor is invoked
  bool already_visited = false;
  EXPAND_PACK_LEFT_TO_RIGHT(
      simple_action_visitor_helper<Invokable, ParallelComponent, Variants>(
          box, &iter, &already_visited, std::forward<Args>(args)...));
}
}  // namespace Algorithm_detail
}  // namespace Parallel
