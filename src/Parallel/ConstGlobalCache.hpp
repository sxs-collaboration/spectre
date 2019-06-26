// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template ConstGlobalCache.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Parallel/CharmRegistration.hpp"
#include "Parallel/ConstGlobalCacheHelper.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

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

// Note: Returned list does not need to be size 1
template <class ConstGlobalCacheTag, class Metavariables>
using get_list_of_matching_tags =
    tmpl::filter<ConstGlobalCache_detail::make_tag_list<Metavariables>,
                 std::is_base_of<tmpl::pin<ConstGlobalCacheTag>, tmpl::_1>>;

template <class ConstGlobalCacheTag, class Metavariables>
using type_for_get = typename type_for_get_helper<
    typename tmpl::front<ConstGlobalCache_detail::get_list_of_matching_tags<
        ConstGlobalCacheTag, Metavariables>>::type>::type;

template <typename ComponentFromList, typename ComponentToFind>
struct get_component_if_mocked_helper
    : std::is_same<typename ComponentFromList::component_being_mocked,
                   ComponentToFind> {};

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
                     get_component_if_mocked_helper<
                         tmpl::_1, tmpl::pin<ParallelComponent>>>>>>;
}  // namespace ConstGlobalCache_detail

/// \ingroup ParallelGroup
/// A Charm++ chare that caches constant data once per Charm++ node.
///
/// Metavariables must define the following metavariables:
///   - const_global_cache_tag_list   typelist of tags of constant data
///   - component_list   typelist of ParallelComponents
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

  explicit ConstGlobalCache(
      tuples::tagged_tuple_from_typelist<
          ConstGlobalCache_detail::make_tag_list<Metavariables>>
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

  tuples::tagged_tuple_from_typelist<
      ConstGlobalCache_detail::make_tag_list<Metavariables>>
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
/// \brief Compute item used to retrieve data from the
/// `Parallel::ConstGlobalCache`.
///
/// The `CacheTag` must have a `container_tag` alias that refers to a
/// `db::SimpleTag`. This container tag can be used to retrieve the data from
/// the DataBox.
template <class CacheTag, class ContainerTag = typename CacheTag::container_tag>
struct FromConstGlobalCache : ContainerTag, db::ComputeTag {
  static_assert(std::is_base_of<db::SimpleTag, ContainerTag>::value,
                "The CacheTag::container_tag must be a simple DataBox tag.");
  template <class Metavariables,
            class CacheTagType =
                ConstGlobalCache_detail::type_for_get<CacheTag, Metavariables>,
            class ContainerTagType = db::item_type<ContainerTag>>
  static const CacheTagType& function(
      const Parallel::ConstGlobalCache<Metavariables>* const& cache) {
    static_assert(cpp17::is_same_v<ContainerTagType, CacheTagType>,
                  "The type of the container tag of the cache tag must match "
                  "the tag in the global cache.");
    return Parallel::get<CacheTag>(*cache);
  }
  using argument_tags = tmpl::list<ConstGlobalCache>;
};
}  // namespace Tags
}  // namespace Parallel

#define CK_TEMPLATES_ONLY
#include "Parallel/ConstGlobalCache.def.h"
#undef CK_TEMPLATES_ONLY
