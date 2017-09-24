// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class template ConstGlobalCache.

#pragma once

#include "ErrorHandling/Assert.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TaggedTuple.hpp"

#include "Parallel/ConstGlobalCache.decl.h"

namespace Parallel {

/// \ingroup Parallel
/// A Charm++ chare that caches constant data once per Charm++ node.
///
/// Metavariables must define the following metavariables:
///   - const_global_cache_tag_list   typelist of tags of constant data
///   - tentacle_list   typelist of Tentacles
template <typename Metavariables>
class ConstGlobalCache : public CBase_ConstGlobalCache<Metavariables> {
 public:
  /// Access to the Metavariables template parameter
  using metavariables = Metavariables;
  /// Typelist of the tags of constant data stored in the ConstGlobalCache
  using tag_list = typename Metavariables::const_global_cache_tag_list;
  /// Typelist of the Tentacles stored in the ConstGlobalCache
  using tentacle_list = typename Metavariables::tentacle_list;

  explicit ConstGlobalCache(
      tuples::TaggedTupleTypelist<tag_list>& const_global_cache) noexcept
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
  /// \brief Access the Charm++ proxy associated with a Tentacle
  ///
  /// \requires TentacleTag is a tag in tentacle_list
  ///
  /// \returns a Charm++ proxy that can be used to call an entry method on the
  /// chare(s)
  template <typename TentacleTag>
  typename TentacleTag::type& get_tentacle() noexcept {
    return tuples::get<TentacleTag>(tentacles_);
  }

  template <typename TentacleTag>
  const typename TentacleTag::type& get_tentacle() const noexcept {
    return tuples::get<TentacleTag>(tentacles_);
  }
  // @}

  /// Entry method to set the Tentacles (should only be called once)
  void set_tentacles(tuples::TaggedTupleTypelist<tentacle_list>& tentacles,
                     const CkCallback& callback) noexcept;

 private:
  tuples::TaggedTupleTypelist<tag_list> const_global_cache_;
  tuples::TaggedTupleTypelist<tentacle_list> tentacles_;
  bool tentacles_have_been_set_{false};
};

template <typename Metavariables>
void ConstGlobalCache<Metavariables>::set_tentacles(
    tuples::TaggedTupleTypelist<typename Metavariables::tentacle_list>&
        tentacles,
    const CkCallback& callback) noexcept {
  ASSERT(!tentacles_have_been_set_, "Can only set the tentacles once");
  tentacles_ = std::move(tentacles);
  tentacles_have_been_set_ = true;
  this->contribute(callback);
}

}  // namespace Parallel

#define CK_TEMPLATES_ONLY
#include "Parallel/ConstGlobalCache.def.h"
#undef CK_TEMPLATES_ONLY
