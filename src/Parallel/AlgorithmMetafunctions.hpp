// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TMPL.hpp"

namespace Parallel::Algorithm_detail {

template <typename Action, typename enable = std::void_t<>>
struct optional_simple_tags {
  using type = tmpl::list<>;
};

template <typename Action>
struct optional_simple_tags<Action, std::void_t<typename Action::simple_tags>> {
  using type = typename Action::simple_tags;
};

template <typename Action, typename enable = std::void_t<>>
struct optional_compute_tags {
  using type = tmpl::list<>;
};

template <typename Action>
struct optional_compute_tags<Action,
                             std::void_t<typename Action::compute_tags>> {
  using type = typename Action::compute_tags;
};

template <typename ActionList>
struct get_action_list_simple_tags {
  using type = tmpl::flatten<tmpl::transform<typename ActionList::action_list,
                                             optional_simple_tags<tmpl::_1>>>;
};

template <typename ActionList>
struct get_action_list_compute_tags {
  using type = tmpl::flatten<tmpl::transform<typename ActionList::action_list,
                                             optional_compute_tags<tmpl::_1>>>;
};

template <typename Pdal>
using get_pdal_simple_tags = tmpl::remove_duplicates<tmpl::flatten<
    tmpl::transform<Pdal, get_action_list_simple_tags<tmpl::_1>>>>;

template <typename Pdal>
using get_pdal_compute_tags = tmpl::remove_duplicates<tmpl::flatten<
    tmpl::transform<Pdal, get_action_list_compute_tags<tmpl::_1>>>>;

template <typename ParallelComponent>
using action_list_simple_tags = get_pdal_simple_tags<
    typename ParallelComponent::phase_dependent_action_list>;

template <typename ParallelComponent>
using action_list_compute_tags = tmpl::remove_duplicates<get_pdal_compute_tags<
    typename ParallelComponent::phase_dependent_action_list>>;
}  // namespace Parallel::Algorithm_detail
