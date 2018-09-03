// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>  // for declval

#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Parallel {
namespace Algorithm_detail {
template <bool, typename AdditionalArgsList>
struct build_action_return_types_impl;

template <typename... AdditionalArgs>
struct build_action_return_types_impl<false, tmpl::list<AdditionalArgs...>> {
  template <typename LastReturnType, typename ReturnTypeList>
  using f = tmpl::push_back<ReturnTypeList, LastReturnType>;
};

template <typename... AdditionalArgs>
struct build_action_return_types_impl<true, tmpl::list<AdditionalArgs...>> {
  template <typename LastReturnType, typename ReturnTypeList, typename Action,
            typename... Actions>
  using f = typename build_action_return_types_impl<
      sizeof...(Actions) != 0, tmpl::list<AdditionalArgs...>>::
      template f<
          std::decay_t<std::tuple_element_t<
              0,
              std::decay_t<decltype(Action::apply(
                  std::declval<std::add_lvalue_reference_t<LastReturnType>>(),
                  std::declval<
                      std::add_lvalue_reference_t<AdditionalArgs>>()...))>>>,
          tmpl::push_back<ReturnTypeList, LastReturnType>, Actions...>;
};

/*!
 * \ingroup ParallelGroup
 * \brief Returns a typelist of the return types of all Actions in ActionList
 *
 * \metareturns
 * typelist
 *
 * \tparam ActionsPack parameter pack of Actions taken
 * \tparam FirstInputParameterType the type of the first argument of the first
 * Action in the ActionsPack
 * \tparam AdditionalArgsList the types of the arguments after the first
 * argument, which must all be the same for all Actions in the ActionsPack
 */
template <typename FirstInputParameterType, typename AdditionalArgsList,
          typename... ActionsPack>
using build_action_return_typelist =
    typename Algorithm_detail::build_action_return_types_impl<
        sizeof...(ActionsPack) != 0, AdditionalArgsList>::
        template f<FirstInputParameterType, tmpl::list<>, ActionsPack...>;

CREATE_IS_CALLABLE(is_ready)

CREATE_IS_CALLABLE(apply)
}  // namespace Algorithm_detail
}  // namespace Parallel
