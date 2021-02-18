// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace observers::Actions {

/// Local synchronous action for retrieving a pointer to the `NodeLock` with tag
/// `LockTag` on the component.
///
/// \warning The retrieved pointer to a lock must only be treated as 'good'
/// during the execution of the action from which this synchronous action is
/// called. This is because we can only trust that charm will not migrate the
/// component on which the action is running until after the action has
/// completed, and it will not migrate the Nodegroup to which the lock points
/// until a checkpoint.
template <typename LockTag>
struct GetLockPointer {
  using return_type = Parallel::NodeLock*;

  template <typename ParallelComponent, typename DbTagList>
  static return_type apply(
      db::DataBox<DbTagList>& box,
      const gsl::not_null<Parallel::NodeLock*> node_lock) noexcept {
    if constexpr (tmpl::list_contains_v<DbTagList, LockTag>) {
      Parallel::NodeLock* result_lock;
      node_lock->lock();
      db::mutate<LockTag>(
          make_not_null(&box),
          [&result_lock](
              const gsl::not_null<Parallel::NodeLock*> lock) noexcept {
            result_lock = lock;
          });
      node_lock->unlock();
      return result_lock;
    } else {
      // silence 'unused variable' warnings
      (void)node_lock;
      ERROR("Could not find required tag " << pretty_type::get_name<LockTag>()
                                           << " in the databox");
    }
  }
};
}  // namespace observers::Actions
