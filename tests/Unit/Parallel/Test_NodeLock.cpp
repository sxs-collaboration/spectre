// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Parallel/NodeLock.hpp"

// [[OutputRegex, Trying to lock a destroyed lock]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Parallel.NodeLock.DestroyedReuseError",
                               "[Parallel][Unit]") {
  ERROR_TEST();
  Parallel::NodeLock lock{};
  lock.destroy();
  lock.lock();
  ERROR("Error not triggered in error test");
}

SPECTRE_TEST_CASE("Unit.Parallel.NodeLock", "[Unit][Parallel]") {
  Parallel::NodeLock first_lock{};
  Parallel::NodeLock second_lock{};
  CHECK(first_lock.try_lock());
  CHECK_FALSE(first_lock.try_lock());
  CHECK(second_lock.try_lock());
  CHECK_FALSE(second_lock.try_lock());
  first_lock.unlock();
  CHECK_FALSE(second_lock.try_lock());
  CHECK(first_lock.try_lock());
  first_lock.destroy();

  CHECK_FALSE(second_lock.try_lock());
  second_lock.unlock();
  CHECK(second_lock.try_lock());
}
