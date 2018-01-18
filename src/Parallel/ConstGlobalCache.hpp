// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template ConstGlobalCache.

#pragma once

#include "ErrorHandling/Assert.hpp"
#include "Parallel/ConstGlobalCacheHelper.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
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
    tmpl::filter<typename ConstGlobalCache<Metavariables>::tag_list,
                 std::is_base_of<ConstGlobalCacheTag, tmpl::_1>>;

template <class ConstGlobalCacheTag, class Metavariables>
using type_for_get = typename type_for_get_helper<
    typename tmpl::front<ConstGlobalCache_detail::get_list_of_matching_tags<
        ConstGlobalCacheTag, Metavariables>>::type>::type;
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
  /// Typelist of the tags of constant data stored in the ConstGlobalCache
  using tag_list = ConstGlobalCache_detail::make_tag_list<Metavariables>;
  /// Typelist of the ParallelComponents stored in the ConstGlobalCache
  using component_list = typename Metavariables::component_list;

  explicit ConstGlobalCache(
      tuples::TaggedTupleTypelist<tag_list> const_global_cache) noexcept
      : const_global_cache_(std::move(const_global_cache)) {}
  explicit ConstGlobalCache(CkMigrateMessage* /*msg*/) {}

  /// Entry method to set the ParallelComponents (should only be called once)
  void set_parallel_components(
      tuples::TaggedTupleTypelist<parallel_component_tag_list>&
          parallel_components,
      const CkCallback& callback) noexcept;

 private:
  // clang-tidy: false positive, redundant declaration
  template <typename ConstGlobalCacheTag, typename MV>
  friend auto get(const ConstGlobalCache<MV>& cache) noexcept  // NOLINT
      -> const ConstGlobalCache_detail::type_for_get<ConstGlobalCacheTag, MV>&;

  // clang-tidy: false positive, redundant declaration
  template <typename ParallelComponentTag, typename MV>
  friend Parallel::proxy_from_parallel_component<ParallelComponentTag>&
  get_parallel_component(ConstGlobalCache<MV>& cache) noexcept;  // NOLINT

  // clang-tidy: false positive, redundant declaration
  template <typename ParallelComponentTag, typename MV>
  friend const Parallel::proxy_from_parallel_component<ParallelComponentTag>&
  get_parallel_component(const ConstGlobalCache<MV>& cache) noexcept;  // NOLINT

  tuples::TaggedTupleTypelist<tag_list> const_global_cache_;
  tuples::TaggedTupleTypelist<parallel_component_tag_list> parallel_components_;
  bool parallel_components_have_been_set_{false};
};

template <typename Metavariables>
void ConstGlobalCache<Metavariables>::set_parallel_components(
    tuples::TaggedTupleTypelist<parallel_component_tag_list>&
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
Parallel::proxy_from_parallel_component<ParallelComponentTag>&
get_parallel_component(ConstGlobalCache<Metavariables>& cache) noexcept {
  return tuples::get<tmpl::type_<
      Parallel::proxy_from_parallel_component<ParallelComponentTag>>>(
      cache.parallel_components_);
}

template <typename ParallelComponentTag, typename Metavariables>
const Parallel::proxy_from_parallel_component<ParallelComponentTag>&
get_parallel_component(const ConstGlobalCache<Metavariables>& cache) noexcept {
  return tuples::get<tmpl::type_<
      Parallel::proxy_from_parallel_component<ParallelComponentTag>>>(
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
}  // namespace Parallel

#define CK_TEMPLATES_ONLY
#include "Parallel/ConstGlobalCache.def.h"
#undef CK_TEMPLATES_ONLY
