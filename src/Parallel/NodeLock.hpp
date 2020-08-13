// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <converse.h>
#include <pup.h>

namespace Parallel {

/*!
 * \ingroup ParallelGroup
 * \brief A typesafe wrapper for a lock for synchronization of shared resources
 * on a given node, with safe creation, destruction, and serialization.
 *
 * \details This structure is a thin wrapper around the charm `CmiNodeLock`, in
 * the <a href="https://charm.readthedocs.io/en/latest/converse/manual.html">
 * Converse library</a>. On construction, this class creates a Converse
 * nodelock, and frees the lock on destruction. When the lock is
 * deserialized, the nodelock is re-created.
 */
class NodeLock {
 public:
  NodeLock() noexcept;

  explicit NodeLock(CkMigrateMessage* /*message*/) noexcept {}

  NodeLock(const NodeLock&) = delete;
  NodeLock& operator=(const NodeLock&) = delete;
  NodeLock(NodeLock&& moved_lock) noexcept;
  NodeLock& operator=(NodeLock&& moved_lock) noexcept;
  ~NodeLock() noexcept;

  void lock() noexcept;

  bool try_lock() noexcept;

  void unlock() noexcept;

  void destroy() noexcept;

  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  bool destroyed_ = false;
  CmiNodeLock lock_{};  // NOLINT
};
}  // namespace Parallel
