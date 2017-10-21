// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace Parallel {
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
        ParallelComponent, typename ParallelComponent::metavariables,
        typename ParallelComponent::action_list,
        typename ParallelComponent::inbox_tag_list,
        typename get_array_index<typename ParallelComponent::chare_type>::
            template f<ParallelComponent>,
        typename ParallelComponent::initial_databox>;

template <typename ParallelComponent>
using index_from_parallel_component =
    typename ParallelComponent::chare_type::template ckindex<
        ParallelComponent, typename ParallelComponent::metavariables,
        typename ParallelComponent::action_list,
        typename ParallelComponent::inbox_tag_list,
        typename get_array_index<typename ParallelComponent::chare_type>::
            template f<ParallelComponent>,
        typename ParallelComponent::initial_databox>;

template <class Metavariables>
struct ConstGlobalCache;

template <class Metavariables>
struct CProxy_ConstGlobalCache;
}  // namespace Parallel
/// \endcond
