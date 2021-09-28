// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>
#include <utility>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"

namespace Parallel {
/// \ingroup ParallelGroup
/// \brief Structs that have `insert_into_inbox` methods for commonly used
/// cases.
///
/// Inbox tags can inherit from the inserters to gain the `insert_into_inbox`
/// static method used by the parallel infrastructure to store data in the
/// inboxes. Collected in the `InboxInserters` namespace are implementations for
/// the most common types of data sent that one may encounter.
namespace InboxInserters {
/*!
 * \brief Inserter for inserting data that is received as the `value_type` (with
 * non-const `key_type`) of a map data structure.
 *
 * An example would be a `std::unordered_map<int, int>`, where the first `int`
 * is the "spatial" id of the parallel component array. The data type that is
 * sent to the component would be a `std::pair<int, int>`, or more generally a
 * `std::pair<key_type, mapped_type>`. The inbox tag would be:
 *
 * \snippet Test_InboxInserters.cpp map tag
 */
template <typename InboxTag>
struct Map {
  template <typename Inbox, typename TemporalId, typename ReceiveDataType>
  static void insert_into_inbox(const gsl::not_null<Inbox*> inbox,
                                const TemporalId& temporal_id,
                                ReceiveDataType&& data) {
    static_assert(std::is_same<std::decay_t<TemporalId>,
                               typename InboxTag::temporal_id>::value,
                  "The temporal_id of the inbox tag does not match the "
                  "received temporal id.");
    auto& current_inbox = (*inbox)[temporal_id];
    ASSERT(0 == current_inbox.count(data.first),
           "Receiving data from the 'same' source twice. The message id is: "
               << data.first);
    if (not current_inbox.insert(std::forward<ReceiveDataType>(data)).second) {
      ERROR("Failed to insert data to receive at instance '"
            << temporal_id << "' with tag '"
            << pretty_type::get_name<InboxTag>() << "'.\n");
    }
  }
};

/*!
 * \brief Inserter for inserting data that is received as the `value_type` of a
 * data structure that has an `insert(value_type&&)` member function.
 *
 * Can be used for receiving data into data structures like
 * `std::(unordered_)multiset`.
 *
 * \snippet Test_InboxInserters.cpp member insert tag
 */
template <typename InboxTag>
struct MemberInsert {
  template <typename Inbox, typename TemporalId, typename ReceiveDataType>
  static void insert_into_inbox(const gsl::not_null<Inbox*> inbox,
                                const TemporalId& temporal_id,
                                ReceiveDataType&& data) {
    static_assert(std::is_same<std::decay_t<TemporalId>,
                               typename InboxTag::temporal_id>::value,
                  "The temporal_id of the inbox tag does not match the "
                  "received temporal id.");
    (*inbox)[temporal_id].insert(std::forward<ReceiveDataType>(data));
  }
};

/*!
 * \brief Inserter for data that is inserted by copy/move.
 *
 * \snippet Test_InboxInserters.cpp value tag
 */
template <typename InboxTag>
struct Value {
  template <typename Inbox, typename TemporalId, typename ReceiveDataType>
  static void insert_into_inbox(const gsl::not_null<Inbox*> inbox,
                                const TemporalId& temporal_id,
                                ReceiveDataType&& data) {
    static_assert(std::is_same<std::decay_t<TemporalId>,
                               typename InboxTag::temporal_id>::value,
                  "The temporal_id of the inbox tag does not match the "
                  "received temporal id.");
    (*inbox)[temporal_id] = std::forward<ReceiveDataType>(data);
  }
};

/*!
 * \brief Inserter for inserting data that is received as the `value_type` of a
 * data structure that has an `push_back(value_type&&)` member function.
 *
 * Can be used for receiving data into data structures like
 * `std::vector`.
 *
 * \snippet Test_InboxInserters.cpp pushback tag
 */
template <typename InboxTag>
struct Pushback {
  template <typename Inbox, typename TemporalId, typename ReceiveDataType>
  static void insert_into_inbox(const gsl::not_null<Inbox*> inbox,
                                const TemporalId& temporal_id,
                                ReceiveDataType&& data) {
    static_assert(std::is_same<std::decay_t<TemporalId>,
                               typename InboxTag::temporal_id>::value,
                  "The temporal_id of the inbox tag does not match the "
                  "received temporal id.");
    (*inbox)[temporal_id].push_back(std::forward<ReceiveDataType>(data));
  }
};
}  // namespace InboxInserters
}  // namespace Parallel
