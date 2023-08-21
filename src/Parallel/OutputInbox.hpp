// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"

namespace Parallel {
namespace detail {
CREATE_IS_CALLABLE(output_inbox)
CREATE_IS_CALLABLE_V(output_inbox)
}  // namespace detail

/*!
 * \brief Returns a string of the contents of the inbox.
 *
 * Calls a static `output_inbox` member function of the `InboxTag`. If the inbox
 * tag doesn't have the `output_inbox` function, then this function can't be
 * called with that inbox (a `static_assert` will fail).
 *
 * \note It's up to the individual inbox what to include in the string. Data may
 * or may not be included
 *
 * \tparam InboxTag Inbox tag with an `output_inbox` member function
 * \tparam InboxTypes Types of all inboxes (deduced)
 * \param inboxes All inboxes
 * \param indent_pad Number of empty spaces to pad the message with incase
 * indenting is needed.
 * \return std::string The contents of the inbox
 */
template <typename InboxTag, typename... InboxTypes>
std::string output_inbox(const tuples::TaggedTuple<InboxTypes...>& inboxes,
                         const size_t indent_pad) {
  static_assert(tmpl::list_contains_v<tmpl::list<InboxTypes...>, InboxTag>);

  return InboxTag::output_inbox(tuples::get<InboxTag>(inboxes), indent_pad);
}
}  // namespace Parallel
