// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template ConstGlobalCache.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Parallel/CharmRegistration.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

#include "Parallel/ConstGlobalCache.decl.h"

namespace Parallel {

namespace ConstGlobalCache_detail {
template <typename T>
struct type_for_get_helper {
  using type = T;
};

template <typename T, typename D>
struct type_for_get_helper<std::unique_ptr<T, D>> {
  using type = T;
};

// This struct provides a better error message if
// an unknown tag is requested from the ConstGlobalCache.
template <typename ConstGlobalCacheTag, typename ListOfPossibleTags>
struct list_of_matching_tags_helper {
  using type = tmpl::filter<ListOfPossibleTags,
               std::is_base_of<tmpl::pin<ConstGlobalCacheTag>, tmpl::_1>>;
  static_assert(not cpp17::is_same_v<type, tmpl::list<>>,
                "Trying to get a nonexistent tag from the ConstGlobalCache. "
                "To diagnose the problem, search for "
                "'list_of_matching_tags_helper' in the error message. "
                "The first template parameter of "
                "'list_of_matching_tags_helper' is the requested tag, and "
                "the second template parameter is a tmpl::list of all the "
                "tags in the ConstGlobalCache.  One possible bug that may "
                "lead to this error message is a missing or misspelled "
                "const_global_cache_tags type alias.");
};

// Note: Returned list does not need to be size 1
template <class ConstGlobalCacheTag, class Metavariables>
using get_list_of_matching_tags = typename list_of_matching_tags_helper<
    ConstGlobalCacheTag, get_const_global_cache_tags<Metavariables>>::type;

template <class ConstGlobalCacheTag, class Metavariables>
using type_for_get = typename type_for_get_helper<
    typename tmpl::front<ConstGlobalCache_detail::get_list_of_matching_tags<
        ConstGlobalCacheTag, Metavariables>>::type>::type;

template <class T, class = cpp17::void_t<>>
struct has_component_being_mocked_alias : std::false_type {};

template <class T>
struct has_component_being_mocked_alias<
    T, cpp17::void_t<typename T::component_being_mocked>> : std::true_type {};

template <class T>
constexpr bool has_component_being_mocked_alias_v =
    has_component_being_mocked_alias<T>::value;

template <typename ComponentToFind, typename ComponentFromList>
struct get_component_if_mocked_helper {
  static_assert(
      has_component_being_mocked_alias_v<ComponentFromList>,
      "The parallel component was not found, and it looks like it is not being "
      "mocked. Did you forget to add it to the "
      "'Metavariables::component_list'? See the first template parameter for "
      "the component that we are looking for and the second template parameter "
      "for the component that is being checked for mocking it.");
  using type = std::is_same<typename ComponentFromList::component_being_mocked,
                            ComponentToFind>;
};

/// In order to be able to use a mock action testing framework we need to be
/// able to get the correct parallel component from the global cache even when
/// the correct component is a mock. We do this by having the mocked components
/// have a member type alias `component_being_mocked`, and having
/// `Parallel::get_component` check if the component to be retrieved is in the
/// `metavariables::component_list`. If it is not in the `component_list` then
/// we search for a mock component that is mocking the component we are trying
/// to retrieve.
template <typename ComponentList, typename ParallelComponent>
using get_component_if_mocked = tmpl::front<tmpl::type_from<tmpl::conditional_t<
    tmpl::list_contains_v<ComponentList, ParallelComponent>,
    tmpl::type_<tmpl::list<ParallelComponent>>,
    tmpl::lazy::find<ComponentList,
                     tmpl::type_<get_component_if_mocked_helper<
                         tmpl::pin<ParallelComponent>, tmpl::_1>>>>>>;
}  // namespace ConstGlobalCache_detail

/// \ingroup ParallelGroup
/// A Charm++ chare that caches constant data once per Charm++ node.
///
/// `Metavariables` must define the following metavariables:
///   - `component_list`   typelist of ParallelComponents
///   - `const_global_cache_tags`   (possibly empty) typelist of tags of
///     constant data
///
/// The tag list for the items added to the ConstGlobalCache is created by
/// combining the following tag lists:
///   - `Metavariables::const_global_cache_tags` which should contain only those
///     tags that cannot be added from the other tag lists below.
///   - `Component::const_global_cache_tags` for each `Component` in
///     `Metavariables::component_list` which should contain the tags needed by
///     any simple actions called on the Component, as well as tags need by the
///     `allocate_array` function of an array component.  The type alias may be
///     omitted for an empty list.
///   - `Action::const_global_cache_tags` for each `Action` in the
///     `phase_dependent_action_list` of each `Component` of
///     `Metavariables::component_list` which should contain the tags needed by
///     that  Action.  The type alias may be omitted for an empty list.
///
/// The tags in the `const_global_cache_tags` type lists are db::SimpleTag%s
/// that have a `using option_tags` type alias and a static function
/// `create_from_options` that are used to create the constant data from input
/// file options.
///
/// References to items in the ConstGlobalCache are also added to the
/// db::DataBox of each `Component` in the `Metavariables::component_list` with
/// the same tag with which they were inserted into the ConstGlobalCache.
template <typename Metavariables>
class ConstGlobalCache : public CBase_ConstGlobalCache<Metavariables> {
  using parallel_component_tag_list = tmpl::transform<
      typename Metavariables::component_list,
      tmpl::bind<
          tmpl::type_,
          tmpl::bind<Parallel::proxy_from_parallel_component, tmpl::_1>>>;

 public:
  /// Access to the Metavariables template parameter
  using metavariables = Metavariables;
  /// Typelist of the ParallelComponents stored in the ConstGlobalCache
  using component_list = typename Metavariables::component_list;

  explicit ConstGlobalCache(tuples::tagged_tuple_from_typelist<
                            get_const_global_cache_tags<Metavariables>>
                                const_global_cache) noexcept
      : const_global_cache_(std::move(const_global_cache)) {}
  explicit ConstGlobalCache(CkMigrateMessage* /*msg*/) {}
  ~ConstGlobalCache() noexcept override {
    (void)Parallel::charmxx::RegisterChare<
        ConstGlobalCache<Metavariables>,
        CkIndex_ConstGlobalCache<Metavariables>>::registrar;
  }
  /// \cond
  ConstGlobalCache(const ConstGlobalCache&) = default;
  ConstGlobalCache& operator=(const ConstGlobalCache&) = default;
  ConstGlobalCache(ConstGlobalCache&&) = default;
  ConstGlobalCache& operator=(ConstGlobalCache&&) = default;
  /// \endcond

  /// Entry method to set the ParallelComponents (should only be called once)
  void set_parallel_components(
      tuples::tagged_tuple_from_typelist<parallel_component_tag_list>&
          parallel_components,
      const CkCallback& callback) noexcept;

 private:
  // clang-tidy: false positive, redundant declaration
  template <typename ConstGlobalCacheTag, typename MV>
  friend auto get(const ConstGlobalCache<MV>& cache) noexcept  // NOLINT
      -> const ConstGlobalCache_detail::type_for_get<ConstGlobalCacheTag, MV>&;

  // clang-tidy: false positive, redundant declaration
  template <typename ParallelComponentTag, typename MV>
  friend auto get_parallel_component(  // NOLINT
      ConstGlobalCache<MV>& cache) noexcept
      -> Parallel::proxy_from_parallel_component<
          ConstGlobalCache_detail::get_component_if_mocked<
              typename MV::component_list, ParallelComponentTag>>&;

  // clang-tidy: false positive, redundant declaration
  template <typename ParallelComponentTag, typename MV>
  friend auto get_parallel_component(  // NOLINT
      const ConstGlobalCache<MV>& cache) noexcept
      -> const Parallel::proxy_from_parallel_component<
          ConstGlobalCache_detail::get_component_if_mocked<
              typename MV::component_list,
              ParallelComponentTag>>&;  // NOLINT

  tuples::tagged_tuple_from_typelist<get_const_global_cache_tags<Metavariables>>
      const_global_cache_;
  tuples::tagged_tuple_from_typelist<parallel_component_tag_list>
      parallel_components_;
  bool parallel_components_have_been_set_{false};
};

template <typename Metavariables>
void ConstGlobalCache<Metavariables>::set_parallel_components(
    tuples::tagged_tuple_from_typelist<parallel_component_tag_list>&
        parallel_components,
    const CkCallback& callback) noexcept {
  ASSERT(!parallel_components_have_been_set_,
         "Can only set the parallel_components once");
  parallel_components_ = std::move(parallel_components);
  parallel_components_have_been_set_ = true;
  this->contribute(callback);
}

// @{
/// \ingroup ParallelGroup
/// \brief Access the Charm++ proxy associated with a ParallelComponent
///
/// \requires ParallelComponentTag is a tag in component_list
///
/// \returns a Charm++ proxy that can be used to call an entry method on the
/// chare(s)
template <typename ParallelComponentTag, typename Metavariables>
auto get_parallel_component(ConstGlobalCache<Metavariables>& cache) noexcept
    -> Parallel::proxy_from_parallel_component<
        ConstGlobalCache_detail::get_component_if_mocked<
            typename Metavariables::component_list, ParallelComponentTag>>& {
  return tuples::get<tmpl::type_<Parallel::proxy_from_parallel_component<
      ConstGlobalCache_detail::get_component_if_mocked<
          typename Metavariables::component_list, ParallelComponentTag>>>>(
      cache.parallel_components_);
}

template <typename ParallelComponentTag, typename Metavariables>
auto get_parallel_component(
    const ConstGlobalCache<Metavariables>& cache) noexcept
    -> const Parallel::proxy_from_parallel_component<
        ConstGlobalCache_detail::get_component_if_mocked<
            typename Metavariables::component_list, ParallelComponentTag>>& {
  return tuples::get<tmpl::type_<Parallel::proxy_from_parallel_component<
      ConstGlobalCache_detail::get_component_if_mocked<
          typename Metavariables::component_list, ParallelComponentTag>>>>(
      cache.parallel_components_);
}
// @}

// @{
/// \ingroup ParallelGroup
/// \brief Access data in the cache
///
/// \requires ConstGlobalCacheTag is a tag in tag_list
///
/// \returns a constant reference to an object in the cache
template <typename ConstGlobalCacheTag, typename Metavariables>
auto get(const ConstGlobalCache<Metavariables>& cache) noexcept -> const
    ConstGlobalCache_detail::type_for_get<ConstGlobalCacheTag, Metavariables>& {
  // We check if the tag is to be retrieved directly or via a base class
  using tag = tmpl::front<ConstGlobalCache_detail::get_list_of_matching_tags<
      ConstGlobalCacheTag, Metavariables>>;
  static_assert(tmpl::size<ConstGlobalCache_detail::get_list_of_matching_tags<
                        ConstGlobalCacheTag, Metavariables>>::value == 1,
                "Found more than one tag matching the ConstGlobalCacheTag "
                "requesting to be retrieved.");
  return make_overloader(
      [](std::true_type /*is_unique_ptr*/, auto&& local_cache)
          -> decltype(
              *(tuples::get<tag>(local_cache.const_global_cache_).get())) {
        return *(
            tuples::get<tag>(local_cache.const_global_cache_)
                .get());
      },
      [](std::false_type /*is_unique_ptr*/, auto&& local_cache)
          -> decltype(tuples::get<tag>(local_cache.const_global_cache_)) {
        return tuples::get<tag>(
            local_cache.const_global_cache_);
      })(typename tt::is_a<std::unique_ptr, typename tag::type>::type{}, cache);
}

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ParallelGroup
/// Tag to retrieve the `Parallel::ConstGlobalCache` from the DataBox.
struct ConstGlobalCache : db::BaseTag {};

template <class Metavariables>
struct ConstGlobalCacheImpl : ConstGlobalCache, db::SimpleTag {
  using type = const Parallel::ConstGlobalCache<Metavariables>*;
  static std::string name() noexcept { return "ConstGlobalCache"; }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ParallelGroup
/// Tag used to retrieve data from the `Parallel::ConstGlobalCache`. This is the
/// recommended way for compute tags to retrieve data out of the global cache.
template <class CacheTag>
struct FromConstGlobalCache : CacheTag, db::ComputeTag {
  static std::string name() noexcept {
    return "FromConstGlobalCache(" + pretty_type::short_name<CacheTag>() + ")";
  }
  template <class Metavariables>
  static const ConstGlobalCache_detail::type_for_get<CacheTag, Metavariables>&
  function(const Parallel::ConstGlobalCache<Metavariables>* const& cache) {
    return Parallel::get<CacheTag>(*cache);
  }
  using argument_tags = tmpl::list<ConstGlobalCache>;
};
}  // namespace Tags
}  // namespace Parallel

#define CK_TEMPLATES_ONLY
#include "Parallel/ConstGlobalCache.def.h"
#undef CK_TEMPLATES_ONLY
