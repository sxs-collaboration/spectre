// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Parallel {
namespace detail {
template <typename Metavariables, typename Tag, typename... OptionTags,
          Requires<Tag::pass_metavariables> = nullptr>
typename Tag::type create_initialization_item_from_options(
    const tuples::TaggedTuple<OptionTags...>& options) {
  return tuples::apply<typename Tag::template option_tags<Metavariables>>(
      [](const auto&... option) {
        return Tag::template create_from_options<Metavariables>(option...);
      },
      options);
}

template <typename Metavariables, typename Tag, typename... OptionTags,
          Requires<not Tag::pass_metavariables> = nullptr>
typename Tag::type create_initialization_item_from_options(
    const tuples::TaggedTuple<OptionTags...>& options) {
  return tuples::apply<typename Tag::option_tags>(
      [](const auto&... option) { return Tag::create_from_options(option...); },
      options);
}
}  // namespace detail

/// \ingroup ParallelGroup
/// \brief Given a list of tags and a tagged tuple containing items
/// created from input options, return a tagged tuple of items constructed
/// by calls to create_from_options for each tag in the list.
template <typename Metavariables, typename... Tags, typename... OptionTags>
tuples::TaggedTuple<Tags...> create_from_options(
    const tuples::TaggedTuple<OptionTags...>& options,
    tmpl::list<Tags...> /*meta*/) {
  return {detail::create_initialization_item_from_options<Metavariables, Tags>(
      options)...};
}
}  // namespace Parallel
