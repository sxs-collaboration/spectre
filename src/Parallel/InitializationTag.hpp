// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/DataBox/TagTraits.hpp"

namespace Parallel {
namespace initialization_tag_detail {
template <template <typename> typename>
constexpr bool is_templated() {
  return true;
}
}  // namespace initialization_tag_detail

/*!
 * \ingroup ParallelGroup
 * Concept for an initialization tag with `pass_metavariables` true.
 */
template <typename Tag>
concept templated_initialization_tag =
    db::simple_tag<Tag> and Tag::pass_metavariables and
    initialization_tag_detail::is_templated<Tag::template option_tags>();

/*!
 * \ingroup ParallelGroup
 * Concept for an initialization tag with `pass_metavariables` false.
 */
template <typename Tag>
concept untemplated_initialization_tag =
    db::simple_tag<Tag> and not Tag::pass_metavariables and
    requires { typename Tag::option_tags; };

/*!
 * \ingroup ParallelGroup
 * Concept for an initialization tag.
 */
template <typename Tag>
concept initialization_tag =
    templated_initialization_tag<Tag> or untemplated_initialization_tag<Tag>;
}  // namespace Parallel
