// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/variant/variant.hpp>

#include "DataStructures/DataBox/DataBox.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Error.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Parallel {
namespace Algorithm_detail {
template <typename Invokable, typename InitialDataBox, typename ThisVariant,
          typename... Variants, typename... Args,
          Requires<is_apply_callable_v<
              Invokable, std::add_lvalue_reference_t<ThisVariant>, Args&&...>> =
              nullptr>
void simple_action_visitor_helper(boost::variant<Variants...>& box,
                                  const gsl::not_null<int*> iter,
                                  const gsl::not_null<bool*> already_visited,
                                  Args&&... args) {
  if (box.which() == *iter and not*already_visited) {
    try {
      make_overloader(
          [&box](std::true_type /*returns_void*/, auto&&... my_args) {
            Invokable::apply(boost::get<ThisVariant>(box),
                             std::forward<Args>(my_args)...);
          },
          [&box](std::false_type /*returns_void*/, auto&&... my_args) {
            box = std::get<0>(Invokable::apply(boost::get<ThisVariant>(box),
                                               std::forward<Args>(my_args)...));
            using return_box_type = decltype(std::get<0>(Invokable::apply(
                boost::get<ThisVariant>(box), std::forward<Args>(my_args)...)));
            static_assert(
                cpp17::is_same_v<std::decay_t<return_box_type>,
                                 InitialDataBox> and
                    cpp17::is_same_v<db::DataBox<tmpl::list<>>, ThisVariant>,
                "A simple action must return either void or take an empty "
                "DataBox and return the initial_databox set in the parallel "
                "component.");
          })(
          typename std::is_same<void, decltype(Invokable::apply(
                                          std::declval<ThisVariant&>(),
                                          std::declval<Args>()...))>::type{},
          std::forward<Args>(args)...);

    } catch (std::exception& e) {
      ERROR("Fatal error: Failed to call single Action '"
            << pretty_type::get_name<Invokable>() << "' on iteration '" << iter
            << "' with DataBox type '" << pretty_type::get_name<ThisVariant>()
            << "'\nThe exception is: '" << e.what() << "'\n");
    }
    *already_visited = true;
  }
  (*iter)++;
}

template <typename Invokable, typename InitialDataBox, typename ThisVariant,
          typename... Variants, typename... Args,
          Requires<not is_apply_callable_v<
              Invokable, std::add_lvalue_reference_t<ThisVariant>, Args&&...>> =
              nullptr>
void simple_action_visitor_helper(boost::variant<Variants...>& box,
                                  const gsl::not_null<int*> iter,
                                  const gsl::not_null<bool*> already_visited,
                                  Args&&... /*args*/) {
  if (box.which() == *iter and not*already_visited) {
    ERROR("\nCannot call apply function of '"
          << pretty_type::get_name<Invokable>() << "' with DataBox type '"
          << pretty_type::get_name<ThisVariant>() << "' and arguments '"
          << pretty_type::get_name<tmpl::list<Args...>>() << "'.\n"
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
 * The primary use case for this is to allow executing a single Action at any
 * point in the Algorithm. The current best-known use case for this is setting
 * up initial data. However, the implementation is generic enough to handle a
 * call at any time that is valid. Here valid is defined as the `apply` function
 * only accesses members of the DataBox that are guaranteed to be present when
 * it is invoked, and returns a DataBox of a type that does not break the
 * Algorithm.
 */
template <typename Invokable, typename InitialDataBox, typename... Variants,
          typename... Args>
void simple_action_visitor(boost::variant<Variants...>& box, Args&&... args) {
  // iter is the current element of the variant in the "for loop"
  int iter = 0;
  // already_visited ensures that only one visitor is invoked
  bool already_visited = false;
  static_cast<void>(std::initializer_list<char>{
      (simple_action_visitor_helper<Invokable, InitialDataBox, Variants>(
           box, &iter, &already_visited, std::forward<Args>(args)...),
       '0')...});
}
}  // namespace Algorithm_detail
}  // namespace Parallel
