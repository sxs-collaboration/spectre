// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <optional>
#include <utility>

#include "DataStructures/LinkedMessageQueue.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct Label1;
struct Label2;

// Can't use "Queue" because Charm defines a type with that name.
template <typename Label>
struct MyQueue {
  // Non-copyable
  using type = std::unique_ptr<double>;
};

void test_id() noexcept {
  const LinkedMessageId<int> id_one_nothing{1, {}};
  const LinkedMessageId<int> id_one_two{1, 2};
  const LinkedMessageId<int> id_two_nothing{2, {}};
  CHECK(id_one_nothing == id_one_nothing);
  CHECK_FALSE(id_one_nothing != id_one_nothing);
  CHECK_FALSE(id_one_nothing == id_one_two);
  CHECK(id_one_nothing != id_one_two);
  CHECK_FALSE(id_one_nothing == id_two_nothing);
  CHECK(id_one_nothing != id_two_nothing);
  CHECK(get_output(id_one_two) == "1 (2)");
}

void test_queue() noexcept {
  LinkedMessageQueue<int, tmpl::list<MyQueue<Label1>, MyQueue<Label2>>> queue{};
  CHECK(not queue.next_ready_id().has_value());

  queue.insert<MyQueue<Label1>>({1, {}}, std::make_unique<double>(1.1));
  CHECK(not queue.next_ready_id().has_value());
  queue.insert<MyQueue<Label2>>({1, {}}, std::make_unique<double>(-1.1));

  CHECK(queue.next_ready_id() == std::optional{1});
  {
    const auto out = queue.extract();
    CHECK(*tuples::get<MyQueue<Label1>>(out) == 1.1);
    CHECK(*tuples::get<MyQueue<Label2>>(out) == -1.1);
  }

  queue.insert<MyQueue<Label2>>({3, {1}}, std::make_unique<double>(-3.3));
  CHECK(not queue.next_ready_id().has_value());
  queue.insert<MyQueue<Label2>>({2, {3}}, std::make_unique<double>(-2.2));
  CHECK(not queue.next_ready_id().has_value());
  queue.insert<MyQueue<Label1>>({2, {3}}, std::make_unique<double>(2.2));

  const auto finish_checks = [](decltype(queue) test_queue) noexcept {
    CHECK(not test_queue.next_ready_id().has_value());
    test_queue.insert<MyQueue<Label1>>({3, {1}}, std::make_unique<double>(3.3));

    CHECK(test_queue.next_ready_id() == std::optional{3});
    {
      const auto out = test_queue.extract();
      CHECK(*tuples::get<MyQueue<Label1>>(out) == 3.3);
      CHECK(*tuples::get<MyQueue<Label2>>(out) == -3.3);
    }

    CHECK(test_queue.next_ready_id() == std::optional{2});
    {
      const auto out = test_queue.extract();
      CHECK(*tuples::get<MyQueue<Label1>>(out) == 2.2);
      CHECK(*tuples::get<MyQueue<Label2>>(out) == -2.2);
    }
  };
  finish_checks(serialize_and_deserialize(queue));
  finish_checks(std::move(queue));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.LinkedMessageQueue",
                  "[Unit][DataStructures]") {
  test_id();
  test_queue();
}

// [[OutputRegex, Received duplicate messages at id 1 and previous id --\.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.LinkedMessageQueue.Duplicate",
    "[Unit][DataStructures]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  LinkedMessageQueue<int, tmpl::list<MyQueue<Label1>, MyQueue<Label2>>> queue{};
  queue.insert<MyQueue<Label1>>({1, {}}, std::make_unique<double>(1.1));
  queue.insert<MyQueue<Label1>>({1, {}}, std::make_unique<double>(1.1));
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif /* SPECTRE_DEBUG */
}

// [[OutputRegex, Received messages with different ids \(1 and 2\) but the same
// previous id \(--\)\.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.LinkedMessageQueue.Inconsistent",
    "[Unit][DataStructures]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  LinkedMessageQueue<int, tmpl::list<MyQueue<Label1>, MyQueue<Label2>>> queue{};
  queue.insert<MyQueue<Label1>>({1, {}}, std::make_unique<double>(1.1));
  queue.insert<MyQueue<Label2>>({2, {}}, std::make_unique<double>(1.1));
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif /* SPECTRE_DEBUG */
}

// [[OutputRegex, Cannot extract before all messages have been received\.]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.DataStructures.LinkedMessageQueue.NotReady",
    "[Unit][DataStructures]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  LinkedMessageQueue<int, tmpl::list<MyQueue<Label1>, MyQueue<Label2>>> queue{};
  queue.insert<MyQueue<Label1>>({1, {}}, std::make_unique<double>(1.1));
  queue.extract();
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif /* SPECTRE_DEBUG */
}
