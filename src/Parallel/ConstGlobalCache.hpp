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

/// \ingroup Parallel
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

  // @{
  /// \brief Access data in the cache
  ///
  /// \requires ConstGlobalCacheTag is a tag in tag_list
  ///
  /// \returns a constant reference to an object in the cache
  template <typename ConstGlobalCacheTag,
            Requires<tt::is_a_v<std::unique_ptr,
                                typename ConstGlobalCacheTag::type>> = nullptr>
  const typename ConstGlobalCacheTag::type::element_type& get() const noexcept {
    return *(tuples::get<ConstGlobalCacheTag>(const_global_cache_).get());
  }

  template <typename ConstGlobalCacheTag,
            Requires<not tt::is_a_v<
                std::unique_ptr, typename ConstGlobalCacheTag::type>> = nullptr>
  const typename ConstGlobalCacheTag::type& get() const noexcept {
    return tuples::get<ConstGlobalCacheTag>(const_global_cache_);
  }
  // @}

  // @{
  /// \brief Access the Charm++ proxy associated with a ParallelComponent
  ///
  /// \requires ParallelComponentTag is a tag in component_list
  ///
  /// \returns a Charm++ proxy that can be used to call an entry method on the
  /// chare(s)
  template <typename ParallelComponentTag>
  Parallel::proxy_from_parallel_component<ParallelComponentTag>&
  get_parallel_component() noexcept {
    return tuples::get<tmpl::type_<
        Parallel::proxy_from_parallel_component<ParallelComponentTag>>>(
        parallel_components_);
  }

  template <typename ParallelComponentTag>
  const Parallel::proxy_from_parallel_component<ParallelComponentTag>&
  get_parallel_component() const noexcept {
    return tuples::get<tmpl::type_<
        Parallel::proxy_from_parallel_component<ParallelComponentTag>>>(
        parallel_components_);
  }
  // @}

  /// Entry method to set the ParallelComponents (should only be called once)
  void set_parallel_components(
      tuples::TaggedTupleTypelist<parallel_component_tag_list>&
          parallel_components,
      const CkCallback& callback) noexcept;

 private:
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

}  // namespace Parallel

#define CK_TEMPLATES_ONLY
#include "Parallel/ConstGlobalCache.def.h"
#undef CK_TEMPLATES_ONLY
