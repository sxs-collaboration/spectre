// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <ostream>
#include <pup.h>

#include "Utilities/Serialization/PupStlCpp17.hpp"

/// \ingroup DataStructuresGroup
/// An identifier for an element in a sequence
///
/// The struct contains an ID and the ID for the previous element of
/// the sequence, if any.  This allows the sequence to be
/// reconstructed even when elements are received out-of-order.
///
/// \see LinkedMessageQueue
template <typename Id>
struct LinkedMessageId {
  Id id;
  std::optional<Id> previous;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    p | id;
    p | previous;
  }
};

template <typename Id>
bool operator==(const LinkedMessageId<Id>& a, const LinkedMessageId<Id>& b) {
  return a.id == b.id and a.previous == b.previous;
}

template <typename Id>
bool operator!=(const LinkedMessageId<Id>& a, const LinkedMessageId<Id>& b) {
  return not(a == b);
}

template <typename Id>
std::ostream& operator<<(std::ostream& s, const LinkedMessageId<Id>& id) {
  return s << id.id << " (" << id.previous << ")";
}

template <typename Id>
struct std::hash<LinkedMessageId<Id>> {
  size_t operator()(const LinkedMessageId<Id>& id) const {
    // For normal use, any particular .id should only appear with one
    // .previous, and vice versa, so hashing only one is fine.
    return std::hash<Id>{}(id.id);
  }
};

template <typename Id>
struct LinkedMessageIdLessComparator {
  using is_transparent = std::true_type;
  bool operator()(const Id& id,
                  const LinkedMessageId<Id>& linked_message_id) const {
    return id < linked_message_id.id;
  }
  bool operator()(const LinkedMessageId<Id>& lhs,
                  const LinkedMessageId<Id>& rhs) const {
    return lhs.id < rhs.id or
           (lhs.id == rhs.id and lhs.previous < rhs.previous);
  }
  bool operator()(const LinkedMessageId<Id>& linked_message_id,
                  const Id& id) const {
    return linked_message_id.id < id;
  }
};

template <typename Id>
bool operator<(const LinkedMessageId<Id>& lhs, const LinkedMessageId<Id>& rhs) {
  return LinkedMessageIdLessComparator<Id>{}(lhs, rhs);
}
