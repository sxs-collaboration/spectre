// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/NodeLock.hpp"

#include <pup.h>

#include "Parallel/Spinlock.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace Parallel {

NodeLock::NodeLock()
    : lock_(std::make_unique<typename decltype(lock_)::element_type>()) {}

NodeLock::NodeLock(CkMigrateMessage* /*message*/) {}

NodeLock::NodeLock(NodeLock&& moved_lock) noexcept
    : lock_(std::move(moved_lock.lock_)) {
  moved_lock.lock_ = nullptr;
}

NodeLock& NodeLock::operator=(NodeLock&& moved_lock) noexcept {
  lock_ = std::move(moved_lock.lock_);
  moved_lock.lock_ = nullptr;
  return *this;
}

NodeLock::~NodeLock() { destroy(); }

void NodeLock::lock() {
  if (UNLIKELY(nullptr == lock_)) {
    ERROR("Trying to lock a destroyed lock");
  }
  lock_->lock();
}

bool NodeLock::try_lock() {
  if (UNLIKELY(nullptr == lock_)) {
    ERROR("Trying to try_lock a destroyed lock");
  }
  return lock_->try_lock();
}

void NodeLock::unlock() {
  if (UNLIKELY(nullptr == lock_)) {
    ERROR("Trying to unlock a destroyed lock");
  }
  lock_->unlock();
}

void NodeLock::destroy() {
  if (nullptr == lock_) {
    return;
  }
  lock_ = nullptr;
}

void NodeLock::pup(PUP::er& p) {  // NOLINT
  bool is_null = (nullptr == lock_);
  p | is_null;
  if (is_null) {
    lock_ = nullptr;
  } else {
    if (p.isUnpacking()) {
      lock_ = std::make_unique<typename decltype(lock_)::element_type>();
    }
  }
}
}  // namespace Parallel
