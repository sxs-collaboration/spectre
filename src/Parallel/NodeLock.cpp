// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/NodeLock.hpp"

#include <converse.h>

#include "ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Parallel {

NodeLock::NodeLock() noexcept { lock_ = CmiCreateLock(); }

NodeLock::NodeLock(NodeLock&& moved_lock) noexcept {
  moved_lock.destroy();
  lock_ = CmiCreateLock();
}

NodeLock& NodeLock::operator=(NodeLock&& moved_lock) noexcept {
  moved_lock.destroy();
  lock_ = CmiCreateLock();
  return *this;
}

NodeLock::~NodeLock() noexcept { destroy(); }

void NodeLock::lock() noexcept {
  if (UNLIKELY(destroyed_)) {
    ERROR("Trying to lock a destroyed lock");
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  CmiLock(lock_);
#pragma GCC diagnostic pop
}

bool NodeLock::try_lock() noexcept {
  if (UNLIKELY(destroyed_)) {
    ERROR("Trying to try_lock a destroyed lock");
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  return CmiTryLock(lock_) == 0;
#pragma GCC diagnostic pop
}

void NodeLock::unlock() noexcept {
  if (UNLIKELY(destroyed_)) {
    ERROR("Trying to unlock a destroyed lock");
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  CmiUnlock(lock_);
#pragma GCC diagnostic pop
}

void NodeLock::destroy() noexcept {
  if (destroyed_) {
    return;
  }
  destroyed_ = true;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  CmiDestroyLock(lock_);
#pragma GCC diagnostic pop
}

void NodeLock::pup(PUP::er& p) noexcept {  // NOLINT
  if (p.isUnpacking()) {
    lock_ = CmiCreateLock();
  }
  // this serialization does nothing on packing or sizing.
  // Generally, the lock should not be used after it has been
  // packed, but it should be preserved in its original state
  // when packed by a checkpointing procedure.
}
}  // namespace Parallel
