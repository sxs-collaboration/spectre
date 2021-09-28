// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <ostream>
#include <pup.h>
#include <unordered_map>
#include <utility>

#include "Parallel/PupStlCpp17.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

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

/// \cond
template <typename Id, typename QueueTags>
class LinkedMessageQueue;
/// \endcond

/// \ingroup DataStructuresGroup
/// A collection of queues of asynchronously received data
///
/// This class is designed to collect asynchronously received
/// messages, possibly from multiple sources, until sufficient data
/// has been received for processing.  Each tag in \p QueueTags
/// defines a queue of messages of type `Tag::type`.
template <typename Id, typename... QueueTags>
class LinkedMessageQueue<Id, tmpl::list<QueueTags...>> {
  static_assert(sizeof...(QueueTags) > 0,
                "Must have at least one message source.");

 public:
  using IdType = Id;

  /// Insert data into a given queue at a given ID.  All queues must
  /// receive data with the same collection of \p id_and_previous, but
  /// are not required to receive them in the same order.
  template <typename Tag>
  void insert(const LinkedMessageId<Id>& id_and_previous,
              typename Tag::type message);

  /// The next ID in the received sequence, if all queues have
  /// received messages at that ID.
  std::optional<Id> next_ready_id() const;

  /// All the messages received at the time indicated by
  /// `next_ready_id()`.  It is an error to call this function if
  /// `next_ready_id()` would return an empty optional.  This function
  /// removes the stored messages from the queues, advancing the
  /// internal sequence ID.
  tuples::TaggedTuple<QueueTags...> extract();

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) {
    p | previous_id_;
    p | messages_;
  }

 private:
  template <typename Tag>
  struct Optional {
    using type = std::optional<typename Tag::type>;
  };

  using OptionalTuple = tuples::TaggedTuple<Optional<QueueTags>...>;

  std::optional<Id> previous_id_{};
  std::unordered_map<std::optional<Id>, std::pair<Id, OptionalTuple>>
      messages_{};
};

template <typename Id, typename... QueueTags>
template <typename Tag>
void LinkedMessageQueue<Id, tmpl::list<QueueTags...>>::insert(
    const LinkedMessageId<Id>& id_and_previous, typename Tag::type message) {
  static_assert((... or std::is_same_v<Tag, QueueTags>), "Unrecognized tag.");
  const auto entry =
      messages_
          .insert({id_and_previous.previous,
                   std::pair{id_and_previous.id, OptionalTuple{}}})
          .first;
  auto& [id, tuple] = entry->second;
  ASSERT(id_and_previous.id == id,
         "Received messages with different ids (" << id << " and "
         << id_and_previous.id << ") but the same previous id ("
         << id_and_previous.previous << ").");
  ASSERT(not tuples::get<Optional<Tag>>(tuple).has_value(),
         "Received duplicate messages at id " << id << " and previous id "
         << id_and_previous.previous << ".");
  tuples::get<Optional<Tag>>(tuple) = std::move(message);
}

template <typename Id, typename... QueueTags>
std::optional<Id>
LinkedMessageQueue<Id, tmpl::list<QueueTags...>>::next_ready_id() const {
  const auto current_entry = messages_.find(previous_id_);
  if (current_entry == messages_.end()) {
    return {};
  }
  const auto& current_value = current_entry->second;
  const auto& [current_id, current_messages] = current_value;
  if ((... and
       tuples::get<Optional<QueueTags>>(current_messages).has_value())) {
    return current_id;
  } else {
    return {};
  }
}

template <typename Id, typename... QueueTags>
tuples::TaggedTuple<QueueTags...>
LinkedMessageQueue<Id, tmpl::list<QueueTags...>>::extract() {
  ASSERT(next_ready_id().has_value(),
         "Cannot extract before all messages have been received.");
  const auto current_entry = messages_.find(previous_id_);
  auto& current_value = current_entry->second;
  auto& [current_id, current_messages] = current_value;
  tuples::TaggedTuple<QueueTags...> result{};
  (void)(..., (tuples::get<QueueTags>(result) = std::move(
                   *tuples::get<Optional<QueueTags>>(current_messages))));
  previous_id_ = current_id;
  messages_.erase(current_entry);
  return result;
}
