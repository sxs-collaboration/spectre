// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/DataBox/TagTraits.hpp"

namespace Parallel {
namespace initialization_tag_detail {
// This check should be implementable without all these structs by
// using SFINAE directly in the concept, but that causes clang 13 to
// segfault.
template <template <typename> typename>
struct templated_check;

template <typename Tag, typename = std::void_t<>>
struct has_templated_option_tags : std::false_type {};

template <typename Tag>
struct has_templated_option_tags<
    Tag, std::void_t<templated_check<Tag::template option_tags>>>
    : std::true_type {};
}  // namespace initialization_tag_detail

/*!
 * \ingroup ParallelGroup
 * Concept for an initialization tag with `pass_metavariables` true.
 */
template <typename Tag>
concept templated_initialization_tag =
    db::simple_tag<Tag> and Tag::pass_metavariables and
    initialization_tag_detail::has_templated_option_tags<Tag>::value;

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
