// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Parallel/Callback.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Parallel {
namespace detail {
template <class Action, class = std::void_t<>>
struct get_inbox_tags_from_action {
  using type = tmpl::list<>;
};

template <class Action>
struct get_inbox_tags_from_action<Action,
                                  std::void_t<typename Action::inbox_tags>> {
  using type = typename Action::inbox_tags;
};
}  // namespace detail

/*!
 * \ingroup ParallelGroup
 * \brief Given a list of Actions, get a list of the unique inbox tags
 */
template <class ActionsList>
using get_inbox_tags = tmpl::remove_duplicates<tmpl::join<tmpl::transform<
    ActionsList, detail::get_inbox_tags_from_action<tmpl::_1>>>>;

namespace detail {
// ParallelStruct is a metavariables, component, or action struct
template <class ParallelStruct, class = std::void_t<>>
struct get_const_global_cache_tags_from_parallel_struct {
  using type = tmpl::list<>;
};

template <class ParallelStruct>
struct get_const_global_cache_tags_from_parallel_struct<
    ParallelStruct,
    std::void_t<typename ParallelStruct::const_global_cache_tags>> {
  using type = typename ParallelStruct::const_global_cache_tags;
};

template <class PhaseDependentActionList>
struct get_const_global_cache_tags_from_pdal {
  using type = tmpl::join<tmpl::transform<
      tmpl::flatten<tmpl::transform<
          PhaseDependentActionList,
          get_action_list_from_phase_dep_action_list<tmpl::_1>>>,
      get_const_global_cache_tags_from_parallel_struct<tmpl::_1>>>;
};

template <class Component>
struct get_const_global_cache_tags_from_component {
  using type = typename get_const_global_cache_tags_from_pdal<
      typename Component::phase_dependent_action_list>::type;
};

template <class ParallelStruct, class = std::void_t<>>
struct get_mutable_global_cache_tags_from_parallel_struct {
  using type = tmpl::list<>;
};

template <class ParallelStruct>
struct get_mutable_global_cache_tags_from_parallel_struct<
    ParallelStruct,
    std::void_t<typename ParallelStruct::mutable_global_cache_tags>> {
  using type = typename ParallelStruct::mutable_global_cache_tags;
};

template <class PhaseDependentActionList>
struct get_mutable_global_cache_tags_from_pdal {
  using type = tmpl::join<tmpl::transform<
      tmpl::flatten<tmpl::transform<
                      PhaseDependentActionList,
          get_action_list_from_phase_dep_action_list<tmpl::_1>>>,
      get_mutable_global_cache_tags_from_parallel_struct<tmpl::_1>>>;
};

template <class Component>
struct get_mutable_global_cache_tags_from_component {
  using type = typename get_mutable_global_cache_tags_from_pdal<
      typename Component::phase_dependent_action_list>::type;
};

}  // namespace detail

/*!
 * \ingroup ParallelGroup
 * \brief Given a list of Actions, get a list of the unique tags specified in
 * the actions' `const_global_cache_tags` aliases.
 */
template <class ActionsList>
using get_const_global_cache_tags_from_actions =
    tmpl::remove_duplicates<tmpl::join<tmpl::transform<
        ActionsList,
        detail::get_const_global_cache_tags_from_parallel_struct<tmpl::_1>>>>;

/*!
 * \ingroup ParallelGroup
 * \brief Given a list of Actions, get a list of the unique tags specified in
 * the actions' `mutable_global_cache_tags` aliases.
 */
template <class ActionsList>
using get_mutable_global_cache_tags_from_actions =
    tmpl::remove_duplicates<tmpl::join<tmpl::transform<
        ActionsList,
        detail::get_mutable_global_cache_tags_from_parallel_struct<tmpl::_1>>>>;

/*!
 * \ingroup ParallelGroup
 * \brief Given the metavariables, get a list of the unique tags that will
 * specify the items in the GlobalCache.
 */
template <typename Metavariables>
using get_const_global_cache_tags =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::list<
        typename detail::get_const_global_cache_tags_from_parallel_struct<
            Metavariables>::type,
        tmpl::transform<
            typename Metavariables::component_list,
            detail::get_const_global_cache_tags_from_parallel_struct<tmpl::_1>>,
        tmpl::transform<
            typename Metavariables::component_list,
            detail::get_const_global_cache_tags_from_component<tmpl::_1>>>>>;

/*!
 * \ingroup ParallelGroup
 * \brief Given the metavariables, get a list of the unique tags that will
 * specify the mutable items in the GlobalCache.
 */
template <typename Metavariables>
using get_mutable_global_cache_tags =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::list<
        typename detail::get_mutable_global_cache_tags_from_parallel_struct<
            Metavariables>::type,
        tmpl::transform<
            typename Metavariables::component_list,
            detail::get_mutable_global_cache_tags_from_parallel_struct<
                tmpl::_1>>,
        tmpl::transform<
            typename Metavariables::component_list,
            detail::get_mutable_global_cache_tags_from_component<tmpl::_1>>>>>;

/*!
 * \ingroup ParallelGroup
 * \brief Check whether a tag is retrievable from the const portion of
 * the global cache.
 */
template <typename Metavariables, typename Tag>
constexpr bool is_in_const_global_cache =
    tmpl::size<tmpl::filter<get_const_global_cache_tags<Metavariables>,
                            std::is_base_of<tmpl::pin<Tag>, tmpl::_1>>>::value >
    0;

/*!
 * \ingroup ParallelGroup
 * \brief Check whether a tag is retrievable from the mutable portion of
 * the global cache.
 */
template <typename Metavariables, typename Tag>
constexpr bool is_in_mutable_global_cache =
    tmpl::size<tmpl::filter<get_mutable_global_cache_tags<Metavariables>,
                            std::is_base_of<tmpl::pin<Tag>, tmpl::_1>>>::value >
    0;

/*!
 * \ingroup ParallelGroup
 * \brief Check whether a tag is retrievable from the global cache.
 */
template <typename Metavariables, typename Tag>
constexpr bool is_in_global_cache =
    is_in_const_global_cache<Metavariables, Tag> or
    is_in_mutable_global_cache<Metavariables, Tag>;

template <typename Tag>
struct MutableCacheTag {
  using tag  = Tag;
  using type =
      std::tuple<typename Tag::type, std::vector<std::unique_ptr<Callback>>>;
};

template <typename Tag>
struct get_mutable_cache_tag {
  using type = MutableCacheTag<Tag>;
};

template <typename Metavariables>
using get_mutable_global_cache_tag_storage =
    tmpl::transform<get_mutable_global_cache_tags<Metavariables>,
                    get_mutable_cache_tag<tmpl::_1>>;

namespace GlobalCache_detail {

template <typename T>
struct type_for_get_helper {
  using type = T;
};

template <typename T, typename D>
struct type_for_get_helper<std::unique_ptr<T, D>> {
  using type = T;
};

// This struct provides a better error message if
// an unknown tag is requested from the GlobalCache.
template <typename GlobalCacheTag, typename ListOfPossibleTags>
struct matching_tag_helper {
  using found_tags = tmpl::filter<ListOfPossibleTags,
               std::is_base_of<tmpl::pin<GlobalCacheTag>, tmpl::_1>>;
  static_assert(not std::is_same_v<found_tags, tmpl::list<>>,
                "Trying to get a nonexistent tag from the GlobalCache. "
                "To diagnose the problem, search for "
                "'matching_tag_helper' in the error message. "
                "The first template parameter of "
                "'matching_tag_helper' is the requested tag, and "
                "the second template parameter is a tmpl::list of all the "
                "tags in the GlobalCache.  One possible bug that may "
                "lead to this error message is a missing or misspelled "
                "const_global_cache_tags or mutable_global_cache_tags "
                "type alias.");
  static_assert(tmpl::size<found_tags>::value == 1,
                "Found more than one matching tag. "
                "To diagnose the problem, search for "
                "'matching_tag_helper' in the error message. "
                "The first template parameter of "
                "'matching_tag_helper' is the requested tag, and "
                "the second template parameter is a tmpl::list of all the "
                "tags in the GlobalCache.");
  using type = tmpl::front<found_tags>;
};

template <class GlobalCacheTag, class Metavariables>
using get_matching_mutable_tag = typename matching_tag_helper<
    GlobalCacheTag, get_mutable_global_cache_tags<Metavariables>>::type;
}  // namespace GlobalCache_detail

namespace detail {
template <typename PhaseAction>
struct get_initialization_actions_list {
  using type = tmpl::list<>;
};

template <typename InitializationActionsList>
struct get_initialization_actions_list<Parallel::PhaseActions<
    Parallel::Phase::Initialization, InitializationActionsList>> {
  using type = InitializationActionsList;
};
}  // namespace detail

/// \ingroup ParallelGroup
/// \brief Given the phase dependent action list, return the list of
/// actions in the Initialization phase (or an empty list if the Initialization
/// phase is absent from the phase dependent action list)
template <typename PhaseDepActionList>
using get_initialization_actions_list = tmpl::flatten<tmpl::transform<
    PhaseDepActionList, detail::get_initialization_actions_list<tmpl::_1>>>;

namespace detail {
template <typename Action, typename = std::void_t<>>
struct get_simple_tags_from_options_from_action {
  using type = tmpl::list<>;
};

template <typename Action>
struct get_simple_tags_from_options_from_action<
    Action, std::void_t<typename Action::simple_tags_from_options>> {
  using type = typename Action::simple_tags_from_options;
};
}  // namespace detail

/// \ingroup ParallelGroup
/// \brief Given a list of initialization actions, returns a list of the
/// unique simple_tags_from_options for all the actions.
template <typename InitializationActionsList>
using get_simple_tags_from_options =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
        InitializationActionsList,
        detail::get_simple_tags_from_options_from_action<tmpl::_1>>>>;

namespace detail {
template <typename SimpleTag, typename Metavariables,
          bool PassMetavariables = SimpleTag::pass_metavariables>
struct get_option_tags_from_simple_tag_impl {
  using type = typename SimpleTag::option_tags;
};
template <typename SimpleTag, typename Metavariables>
struct get_option_tags_from_simple_tag_impl<SimpleTag, Metavariables, true> {
  using type = typename SimpleTag::template option_tags<Metavariables>;
};
template <typename Metavariables>
struct get_option_tags_from_simple_tag {
  template <typename SimpleTag>
  using f = tmpl::type_from<
      get_option_tags_from_simple_tag_impl<SimpleTag, Metavariables>>;
};
}  // namespace detail

/// \ingroup ParallelGroup
/// \brief Given a list of simple tags, returns a list of the
/// unique option tags required to construct them.
template <typename SimpleTagsList, typename Metavariables>
using get_option_tags = tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
    SimpleTagsList, tmpl::bind<detail::get_option_tags_from_simple_tag<
                                   Metavariables>::template f,
                               tmpl::_1>>>>;

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
  using f = int;
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
  using cproxy_section =
      typename ParallelComponent::chare_type::template cproxy_section<
          ParallelComponent, Args...>;
};
/// \endcond
}  // namespace Parallel
