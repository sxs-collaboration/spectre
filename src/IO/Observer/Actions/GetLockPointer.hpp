// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <mutex>

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
  static return_type apply(db::DataBox<DbTagList>& box,
                           const gsl::not_null<Parallel::NodeLock*> node_lock) {
    Parallel::NodeLock* result_lock;
    const std::lock_guard hold_lock(*node_lock);
    db::mutate<LockTag>(
        [&result_lock](const gsl::not_null<Parallel::NodeLock*> lock) {
          result_lock = lock;
        },
        make_not_null(&box));
    return result_lock;
  }
};
}  // namespace observers::Actions
