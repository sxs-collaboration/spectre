// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/LinkedMessageQueue.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
/// \endcond

namespace Actions {
/// \ingroup ActionsGroup
/// \brief Add data to a LinkedMessageQueue
///
/// Add the passed `id_and_previous` and `message` to the queue
/// `QueueTag` in the `LinkedMessageQueue` stored in the DataBox tag
/// `LinkedMessageQueueTag`.  Then, for each ID for which the message
/// queue has all required messages, call `Processor::apply`.  The
/// signature for an `apply` function for a message queue containing
/// `Queue1` and `Queue2` with ID type `int` is:
///
/// \snippet Test_UpdateMessageQueue.cpp Processor::apply
template <typename QueueTag, typename LinkedMessageQueueTag, typename Processor>
struct UpdateMessageQueue {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(
      db::DataBox<DbTags>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index,
      const LinkedMessageId<typename LinkedMessageQueueTag::type::IdType>&
          id_and_previous,
      typename QueueTag::type message) noexcept {
    if constexpr (db::tag_is_retrievable_v<LinkedMessageQueueTag,
                                           db::DataBox<DbTags>>) {
      auto& queue =
          db::get_mutable_reference<LinkedMessageQueueTag>(make_not_null(&box));
      queue.template insert<QueueTag>(id_and_previous, std::move(message));
      while (auto id = queue.next_ready_id()) {
        Processor::apply(make_not_null(&box), cache, array_index, *id,
                         queue.extract());
      }
    } else {
      (void)box;
      (void)cache;
      (void)array_index;
      (void)id_and_previous;
      (void)message;
      ERROR("LinkedMessageQueueTag not found in DataBox");  // LCOV_EXCL_LINE
    }
  }
};
}  // namespace Actions
