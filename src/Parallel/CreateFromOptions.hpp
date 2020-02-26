// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Parallel {
namespace detail {
template <typename Metavariables, typename Tag, typename... OptionTags,
          typename... OptionTagsForTag,
          Requires<Tag::pass_metavariables> = nullptr>
typename Tag::type create_initialization_item_from_options(
    const tuples::TaggedTuple<OptionTags...>& options,
    tmpl::list<OptionTagsForTag...> /*meta*/) noexcept {
  return Tag::template create_from_options<Metavariables>(
      tuples::get<OptionTagsForTag>(options)...);
}

template <typename Metavariables, typename Tag, typename... OptionTags,
          typename... OptionTagsForTag,
          Requires<not Tag::pass_metavariables> = nullptr>
typename Tag::type create_initialization_item_from_options(
    const tuples::TaggedTuple<OptionTags...>& options,
    tmpl::list<OptionTagsForTag...> /*meta*/) noexcept {
  return Tag::create_from_options(tuples::get<OptionTagsForTag>(options)...);
}
}  // namespace detail

/// \ingroup ParallelGroup
/// \brief Given a list of tags and a tagged tuple containing items
/// created from input options, return a tagged tuple of items constructed
/// by calls to create_from_options for each tag in the list.
template <typename Metavariables, typename... Tags, typename... OptionTags>
tuples::TaggedTuple<Tags...> create_from_options(
    const tuples::TaggedTuple<OptionTags...>& options,
    tmpl::list<Tags...> /*meta*/) noexcept {
  return {detail::create_initialization_item_from_options<Metavariables, Tags>(
      options, typename Tags::option_tags{})...};
}
}  // namespace Parallel
