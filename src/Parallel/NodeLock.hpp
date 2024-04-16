// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <memory>

#include "Parallel/Spinlock.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Parallel {

/*!
 * \ingroup ParallelGroup
 * \brief A typesafe wrapper for a lock for synchronization of shared resources
 * on a given node, with safe creation, destruction, and serialization.
 *
 * \note If a locked NodeLock is serialized, it is deserialized as unlocked.
 */
class NodeLock {
 public:
  NodeLock();

  explicit NodeLock(CkMigrateMessage* /*message*/);

  NodeLock(const NodeLock&) = delete;
  NodeLock& operator=(const NodeLock&) = delete;
  NodeLock(NodeLock&& moved_lock) noexcept;
  NodeLock& operator=(NodeLock&& moved_lock) noexcept;
  ~NodeLock();

  void lock();

  bool try_lock();

  void unlock();

  void destroy();

  bool is_destroyed() { return nullptr == lock_; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  std::unique_ptr<Spinlock> lock_;
};
}  // namespace Parallel
