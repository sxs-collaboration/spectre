// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestHelpers.hpp"
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

namespace {
void test_two_locks() noexcept {
  Parallel::NodeLock first{};
  Parallel::NodeLock second{};
  CHECK_FALSE(first.is_destroyed());
  CHECK_FALSE(second.is_destroyed());

  CHECK(first.try_lock());
  CHECK_FALSE(first.try_lock());

  CHECK(second.try_lock());
  CHECK_FALSE(second.try_lock());

  first.unlock();
  CHECK_FALSE(second.try_lock());
  CHECK(first.try_lock());

  first.destroy();
  CHECK(first.is_destroyed());
  CHECK_FALSE(second.is_destroyed());
  CHECK_FALSE(second.try_lock());

  second.unlock();
  second.lock();
  CHECK_FALSE(second.try_lock());
}

void test_move_semantics() noexcept {
  Parallel::NodeLock unlocked{};
  Parallel::NodeLock move_of_unlocked(std::move(unlocked));
  CHECK(move_of_unlocked.try_lock());
  CHECK_FALSE(move_of_unlocked.try_lock());
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  CHECK(unlocked.is_destroyed());
  CHECK_FALSE(move_of_unlocked.is_destroyed());

  Parallel::NodeLock locked{};
  locked.lock();
  CHECK_FALSE(locked.try_lock());
  Parallel::NodeLock move_of_locked(std::move(locked));
  CHECK_FALSE(move_of_locked.try_lock());
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  CHECK(locked.is_destroyed());
  CHECK_FALSE(move_of_locked.is_destroyed());

  Parallel::NodeLock destroyed{};
  destroyed.destroy();
  CHECK(destroyed.is_destroyed());
  Parallel::NodeLock move_of_destroyed(std::move(destroyed));
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  CHECK(destroyed.is_destroyed());
  CHECK(move_of_destroyed.is_destroyed());
}

void test_move_assign_semantics() noexcept {
  Parallel::NodeLock unlocked_1{};
  Parallel::NodeLock move_assign_from_unlocked_to_unlocked{};
  move_assign_from_unlocked_to_unlocked = std::move(unlocked_1);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  CHECK(unlocked_1.is_destroyed());
  CHECK_FALSE(move_assign_from_unlocked_to_unlocked.is_destroyed());
  CHECK(move_assign_from_unlocked_to_unlocked.try_lock());

  Parallel::NodeLock unlocked_2{};
  Parallel::NodeLock move_assign_from_unlocked_to_locked{};
  move_assign_from_unlocked_to_locked.lock();
  move_assign_from_unlocked_to_locked = std::move(unlocked_2);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  CHECK(unlocked_2.is_destroyed());
  CHECK_FALSE(move_assign_from_unlocked_to_locked.is_destroyed());
  CHECK(move_assign_from_unlocked_to_locked.try_lock());

  Parallel::NodeLock unlocked_3{};
  Parallel::NodeLock move_assign_from_unlocked_to_destroyed{};
  move_assign_from_unlocked_to_destroyed.destroy();
  move_assign_from_unlocked_to_destroyed = std::move(unlocked_3);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  CHECK(unlocked_3.is_destroyed());
  CHECK_FALSE(move_assign_from_unlocked_to_destroyed.is_destroyed());
  CHECK(move_assign_from_unlocked_to_destroyed.try_lock());

  Parallel::NodeLock locked_1{};
  locked_1.lock();
  Parallel::NodeLock move_assign_from_locked_to_unlocked{};
  move_assign_from_locked_to_unlocked = std::move(locked_1);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  CHECK(locked_1.is_destroyed());
  CHECK_FALSE(move_assign_from_locked_to_unlocked.is_destroyed());
  CHECK_FALSE(move_assign_from_locked_to_unlocked.try_lock());

  Parallel::NodeLock locked_2{};
  locked_2.lock();
  Parallel::NodeLock move_assign_from_locked_to_locked{};
  move_assign_from_locked_to_locked.lock();
  move_assign_from_locked_to_locked = std::move(locked_2);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  CHECK(locked_2.is_destroyed());
  CHECK_FALSE(move_assign_from_locked_to_locked.is_destroyed());
  CHECK_FALSE(move_assign_from_locked_to_locked.try_lock());

  Parallel::NodeLock locked_3{};
  locked_3.lock();
  Parallel::NodeLock move_assign_from_locked_to_destroyed{};
  move_assign_from_locked_to_destroyed.destroy();
  move_assign_from_locked_to_destroyed = std::move(locked_3);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  CHECK(locked_3.is_destroyed());
  CHECK_FALSE(move_assign_from_locked_to_destroyed.is_destroyed());
  CHECK_FALSE(move_assign_from_locked_to_destroyed.try_lock());

  Parallel::NodeLock destroyed_1{};
  destroyed_1.destroy();
  Parallel::NodeLock move_assign_from_destroyed_to_unlocked{};
  move_assign_from_destroyed_to_unlocked = std::move(destroyed_1);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  CHECK(destroyed_1.is_destroyed());
  CHECK(move_assign_from_destroyed_to_unlocked.is_destroyed());

  Parallel::NodeLock destroyed_2{};
  destroyed_2.destroy();
  Parallel::NodeLock move_assign_from_destroyed_to_locked{};
  move_assign_from_destroyed_to_locked.lock();
  move_assign_from_destroyed_to_locked = std::move(destroyed_2);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  CHECK(destroyed_2.is_destroyed());
  CHECK(move_assign_from_destroyed_to_locked.is_destroyed());

  Parallel::NodeLock destroyed_3{};
  destroyed_3.destroy();
  Parallel::NodeLock move_assign_from_destroyed_to_destroyed{};
  move_assign_from_destroyed_to_destroyed.destroy();
  move_assign_from_destroyed_to_destroyed = std::move(destroyed_3);
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move,bugprone-use-after-move)
  CHECK(destroyed_3.is_destroyed());
  CHECK(move_assign_from_destroyed_to_destroyed.is_destroyed());
}

void test_serialization() noexcept {
  Parallel::NodeLock unlocked{};
  Parallel::NodeLock serialized_unlocked = serialize_and_deserialize(unlocked);
  CHECK_FALSE(unlocked.is_destroyed());
  CHECK_FALSE(serialized_unlocked.is_destroyed());
  CHECK(unlocked.try_lock());
  CHECK(serialized_unlocked.try_lock());

  Parallel::NodeLock locked{};
  locked.lock();
  Parallel::NodeLock serialized_locked = serialize_and_deserialize(locked);
  CHECK_FALSE(locked.is_destroyed());
  CHECK_FALSE(serialized_locked.is_destroyed());
  CHECK_FALSE(locked.try_lock());
  // a locked NodeLock is deserialized unlocked
  CHECK(serialized_locked.try_lock());

  Parallel::NodeLock destroyed{};
  destroyed.destroy();
  Parallel::NodeLock serialized_destroyed =
      serialize_and_deserialize(destroyed);
  CHECK(destroyed.is_destroyed());
  CHECK(serialized_destroyed.is_destroyed());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.NodeLock", "[Unit][Parallel]") {
  test_two_locks();
  test_move_semantics();
  test_move_assign_semantics();
  test_serialization();
}
