// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Parallel {
namespace Parallel_detail {
template <class Action, class = cpp17::void_t<>>
struct get_inbox_tags_from_action {
  using type = tmpl::list<>;
};

template <class Action>
struct get_inbox_tags_from_action<Action,
                                  cpp17::void_t<typename Action::inbox_tags>> {
  using type = typename Action::inbox_tags;
};
}  // namespace Parallel_detail

/*!
 * \ingroup ParallelGroup
 * \brief Given an Action returns the list of inbox tags for that action
 */
template <class Action>
using get_inbox_tags_from_action =
    typename Parallel_detail::get_inbox_tags_from_action<Action>::type;

/*!
 * \ingroup ParallelGroup
 * \brief Given a list of Actions, get a list of the unique inbox tags
 */
template <class ActionsList>
using get_inbox_tags = tmpl::remove_duplicates<tmpl::join<tmpl::transform<
    ActionsList, Parallel_detail::get_inbox_tags_from_action<tmpl::_1>>>>;

namespace Parallel_detail {
template <class Action, class = cpp17::void_t<>>
struct get_const_global_cache_tags_from_action {
  using type = tmpl::list<>;
};

template <class Action>
struct get_const_global_cache_tags_from_action<
    Action, cpp17::void_t<typename Action::const_global_cache_tags>> {
  using type = typename Action::const_global_cache_tags;
};
}  // namespace Parallel_detail

/*!
 * \ingroup ParallelGroup
 * \brief Given an Action returns the contents of the
 * `const_global_cache_tags` alias for that action, or an empty list
 * if no such alias exists.
 */
template <class Action>
using get_const_global_cache_tags_from_action =
    typename Parallel_detail::get_const_global_cache_tags_from_action<
        Action>::type;

/*!
 * \ingroup ParallelGroup
 * \brief Given a list of Actions, get a list of the unique tags
 * specified in the actions' `const_global_cache_tags` aliases.
 */
template <class ActionsList>
using get_const_global_cache_tags =
    tmpl::remove_duplicates<tmpl::join<tmpl::transform<
        ActionsList,
        Parallel_detail::get_const_global_cache_tags_from_action<tmpl::_1>>>>;

/// \cond
namespace Algorithms {
struct Singleton;
struct Array;
struct Group;
struct Nodegroup;
}  // namespace Algorithms

template <class ChareType>
struct get_array_index;

template <>
struct get_array_index<Parallel::Algorithms::Singleton> {
  template <class ParallelComponent>
  using f = cpp17::void_type;
};

template <>
struct get_array_index<Parallel::Algorithms::Array> {
  template <class ParallelComponent>
  using f = typename ParallelComponent::array_index;
};

template <>
struct get_array_index<Parallel::Algorithms::Group> {
  template <class ParallelComponent>
  using f = int;
};

template <>
struct get_array_index<Parallel::Algorithms::Nodegroup> {
  template <class ParallelComponent>
  using f = int;
};

template <typename ParallelComponent>
using proxy_from_parallel_component =
    typename ParallelComponent::chare_type::template cproxy<
        ParallelComponent,
        typename get_array_index<typename ParallelComponent::chare_type>::
            template f<ParallelComponent>>;

template <typename ParallelComponent>
using index_from_parallel_component =
    typename ParallelComponent::chare_type::template ckindex<
        ParallelComponent,
        typename get_array_index<typename ParallelComponent::chare_type>::
            template f<ParallelComponent>>;

template <class ParallelComponent, class... Args>
struct charm_types_with_parameters {
  using cproxy =
      typename ParallelComponent::chare_type::template cproxy<ParallelComponent,
                                                              Args...>;
  using cbase =
      typename ParallelComponent::chare_type::template cbase<ParallelComponent,
                                                             Args...>;
  using algorithm =
      typename ParallelComponent::chare_type::template algorithm_type<
          ParallelComponent, Args...>;
  using ckindex = typename ParallelComponent::chare_type::template ckindex<
      ParallelComponent, Args...>;
};
/// \endcond
}  // namespace Parallel
