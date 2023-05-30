// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Parallel/StaticSpscQueue.hpp"

SPECTRE_TEST_CASE("Unit.Parallel.StaticSpscQueue", "[Unit][Parallel]") {
  // We can only test basic functionality, it's difficult to test proper
  // threadsafety since that requires generating a race condition.
  Parallel::StaticSpscQueue<int, 5> queue{};
  CHECK(queue.empty());
  CHECK(queue.size() == 0);  // NOLINT
  CHECK(queue.capacity() == 5);
  queue.emplace(3);
  CHECK_FALSE(queue.empty());
  CHECK(queue.size() == 1);
  CHECK(queue.capacity() == 5);
  queue.push(5);
  CHECK_FALSE(queue.empty());
  CHECK(queue.size() == 2);
  CHECK(queue.capacity() == 5);
  const int a = 7;
  queue.push(a);
  CHECK_FALSE(queue.empty());
  CHECK(queue.size() == 3);
  CHECK(queue.capacity() == 5);

  CHECK(queue.try_emplace(11));
  CHECK_FALSE(queue.empty());
  CHECK(queue.size() == 4);
  CHECK(queue.capacity() == 5);

  CHECK(queue.try_push(15));
  CHECK_FALSE(queue.empty());
  CHECK(queue.size() == 5);
  CHECK(queue.capacity() == 5);

  CHECK_FALSE(queue.try_push(a));
  CHECK_FALSE(queue.empty());
  CHECK(queue.size() == 5);
  CHECK(queue.capacity() == 5);

  CHECK_FALSE(queue.try_push(19));
  CHECK_FALSE(queue.empty());
  CHECK(queue.size() == 5);
  CHECK(queue.capacity() == 5);

  CHECK_FALSE(queue.try_emplace(21));
  CHECK_FALSE(queue.empty());
  CHECK(queue.size() == 5);
  CHECK(queue.capacity() == 5);

  int* front = queue.front();
  REQUIRE(front != nullptr);
  REQUIRE(*front == 3);
  CHECK_FALSE(queue.empty());
  CHECK(queue.size() == 5);
  CHECK(queue.capacity() == 5);
  queue.pop();
  CHECK(queue.size() == 4);

  front = queue.front();
  REQUIRE(front != nullptr);
  REQUIRE(*front == 5);
  CHECK_FALSE(queue.empty());
  CHECK(queue.size() == 4);
  CHECK(queue.capacity() == 5);
  queue.pop();
  CHECK(queue.size() == 3);

  front = queue.front();
  REQUIRE(front != nullptr);
  REQUIRE(*front == 7);
  CHECK_FALSE(queue.empty());
  CHECK(queue.size() == 3);
  CHECK(queue.capacity() == 5);
  queue.pop();
  CHECK(queue.size() == 2);

  front = queue.front();
  REQUIRE(front != nullptr);
  REQUIRE(*front == 11);
  CHECK_FALSE(queue.empty());
  CHECK(queue.size() == 2);
  CHECK(queue.capacity() == 5);
  queue.pop();
  CHECK(queue.size() == 1);

  front = queue.front();
  REQUIRE(front != nullptr);
  REQUIRE(*front == 15);
  CHECK_FALSE(queue.empty());
  CHECK(queue.size() == 1);
  CHECK(queue.capacity() == 5);
  queue.pop();
  CHECK(queue.empty());

  front = queue.front();
  REQUIRE(front == nullptr);
#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(queue.pop(),
                    Catch::Matchers::ContainsSubstring(
                        "Can't pop an element from an empty queue."));

#endif  // SPECTRE_DEBUG
}
