// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Parallel {
/// Extract the `InboxTag` from the `inboxes` at the current temporal ID. The
/// value is returned and erased from the inbox.
template <typename InboxTag, typename TemporalIdTag, typename DbTagsList,
          typename... InboxTags>
typename InboxTag::type::mapped_type extract_from_inbox(
    tuples::TaggedTuple<InboxTags...>& inboxes,
    const db::DataBox<DbTagsList>& box) noexcept {
  return std::move(tuples::get<InboxTag>(inboxes)
                       .extract(db::get<TemporalIdTag>(box))
                       .mapped());
}
}  // namespace Parallel
