// Distributed under the MIT License.
// See LICENSE.txt for details.

// This stuff needs to be in a separate file so it can be used in
// ConstGlobalCache.ci

#pragma once

#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Parallel {
template <typename Tag>
struct GlobalCacheTag {
  static constexpr auto help = Tag::help;
  using type = typename Tag::const_global_cache_type;
};

namespace ConstGlobalCache_detail {
template <typename Tag, typename = cpp17::void_t<>>
struct GlobalCacheTagsImpl {
  using type = Tag;
};
template <typename Tag>
struct GlobalCacheTagsImpl<
    Tag, cpp17::void_t<typename Tag::const_global_cache_type>> {
  using type = GlobalCacheTag<Tag>;
};

template <typename Tag>
using GlobalCacheTagsImpl_t = typename GlobalCacheTagsImpl<Tag>::type;

template <typename Component>
using parallel_component_cache_tags =
    typename Component::const_global_cache_tag_list;

template <typename Metavariables>
using make_option_tag_list = tmpl::remove_duplicates<
    tmpl::append<typename Metavariables::const_global_cache_tag_list,
                 tmpl::join<tmpl::transform<
                     typename Metavariables::component_list,
                     tmpl::bind<parallel_component_cache_tags, tmpl::_1>>>>>;

template <typename Metavariables>
using make_tag_list =
    tmpl::transform<make_option_tag_list<Metavariables>,
                    ConstGlobalCache_detail::GlobalCacheTagsImpl<tmpl::_1>>;
}  // namespace ConstGlobalCache_detail
}  // namespace Parallel
