// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/NodeLock.hpp"

#include <converse.h>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Parallel {

NodeLock::NodeLock() noexcept
    : lock_(std::make_unique<CmiNodeLock>(CmiCreateLock())) {}

NodeLock::NodeLock(NodeLock&& moved_lock) noexcept
    : lock_(std::move(moved_lock.lock_)) {
  moved_lock.lock_ = nullptr;
}

NodeLock& NodeLock::operator=(NodeLock&& moved_lock) noexcept {
  lock_ = std::move(moved_lock.lock_);
  moved_lock.lock_ = nullptr;
  return *this;
}

NodeLock::~NodeLock() noexcept { destroy(); }

void NodeLock::lock() noexcept {
  if (UNLIKELY(nullptr == lock_)) {
    ERROR("Trying to lock a destroyed lock");
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  CmiLock(*lock_);
#pragma GCC diagnostic pop
}

bool NodeLock::try_lock() noexcept {
  if (UNLIKELY(nullptr == lock_)) {
    ERROR("Trying to try_lock a destroyed lock");
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  return CmiTryLock(*lock_) == 0;
#pragma GCC diagnostic pop
}

void NodeLock::unlock() noexcept {
  if (UNLIKELY(nullptr == lock_)) {
    ERROR("Trying to unlock a destroyed lock");
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  CmiUnlock(*lock_);
#pragma GCC diagnostic pop
}

void NodeLock::destroy() noexcept {
  if (nullptr == lock_) {
    return;
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  CmiDestroyLock(*lock_);
#pragma GCC diagnostic pop
  lock_ = nullptr;
}

void NodeLock::pup(PUP::er& p) noexcept {  // NOLINT
  bool is_null = (nullptr == lock_);
  p | is_null;
  if (is_null) {
    lock_ = nullptr;
  } else {
    if (p.isUnpacking()) {
      lock_ = std::make_unique<CmiNodeLock>(CmiCreateLock());
    }
  }
}
}  // namespace Parallel
