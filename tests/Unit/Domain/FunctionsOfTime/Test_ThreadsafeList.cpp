// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <utility>

#include "Domain/FunctionsOfTime/ThreadsafeList.hpp"
#include "Domain/FunctionsOfTime/ThreadsafeList.tpp"
#include "Framework/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.ThreadsafeList",
                  "[Unit][Domain]") {
  using domain::FunctionsOfTime::FunctionOfTimeHelpers::ThreadsafeList;
  ThreadsafeList<int> list(1.0);
  CHECK(list.initial_time() == 1.0);
  CHECK(list.expiration_time() == 1.0);

  CHECK(list == ThreadsafeList<int>(1.0));
  CHECK_FALSE(list != ThreadsafeList<int>(1.0));
  CHECK(list != ThreadsafeList<int>(2.0));
  CHECK_FALSE(list == ThreadsafeList<int>(2.0));

  test_copy_semantics(list);
  test_serialization(list);

  list.insert(1.0, 5, 3.0);
  CHECK(list.initial_time() == 1.0);
  CHECK(list.expiration_time() == 3.0);
  CHECK(list.expiration_after(1.0) == 3.0);
  CHECK(list.expiration_after(2.0) == 3.0);
  {
    const auto entry = list(2.0);
    CHECK(entry.update == 1.0);
    CHECK(entry.data == 5);
    CHECK(entry.expiration == 3.0);
    CHECK(list(1.0) == entry);
    CHECK(list(3.0) == entry);
  }
  CHECK(list != ThreadsafeList<int>(1.0));
  CHECK_FALSE(list == ThreadsafeList<int>(1.0));

  test_copy_semantics(list);

  {
    ThreadsafeList<int> list2(1.0);
    ThreadsafeList<int> list3(1.0);
    list2.insert(1.0, 5, 4.0);
    list3.insert(1.0, 4, 3.0);
    CHECK(list != list2);
    CHECK_FALSE(list == list2);
    CHECK(list != list3);
    CHECK_FALSE(list == list3);
  }

  ThreadsafeList<int> list_copy{};
  {
    auto list_temp1 = serialize_and_deserialize(list);
    CHECK(list_temp1 == list);
    CHECK_FALSE(list_temp1 != list);
    auto list_temp2 = std::move(list_temp1);
    CHECK(list_temp2 == list);
    CHECK_FALSE(list_temp2 != list);
    list_copy = std::move(list_temp2);
    CHECK(list_copy == list);
    CHECK_FALSE(list_copy != list);
  }

  list.insert(3.0, 7, 5.0);
  CHECK(list.initial_time() == 1.0);
  CHECK(list.expiration_time() == 5.0);
  CHECK(list.expiration_after(1.0) == 3.0);
  CHECK(list.expiration_after(2.0) == 3.0);
  CHECK(list.expiration_after(3.0) == 5.0);
  CHECK(list.expiration_after(4.0) == 5.0);
  {
    const auto entry = list(2.0);
    CHECK(entry.update == 1.0);
    CHECK(entry.data == 5);
    CHECK(entry.expiration == 3.0);
    CHECK(list(1.0) == entry);
    CHECK(list(3.0) == entry);
  }
  {
    const auto entry = list(4.0);
    CHECK(entry.update == 3.0);
    CHECK(entry.data == 7);
    CHECK(entry.expiration == 5.0);
    CHECK(list(5.0) == entry);
  }
  CHECK(list(list.initial_time()) ==
        list(list.initial_time() * (1.0 - 1.0e-15)));

  CHECK(list != ThreadsafeList<int>(1.0));
  CHECK_FALSE(list == ThreadsafeList<int>(1.0));
  CHECK(list != list_copy);
  CHECK_FALSE(list == list_copy);

  test_copy_semantics(list);

  {
    const ThreadsafeList<int> list2(1.0);
    CHECK(list2.begin() == list2.end());
    CHECK_FALSE(list2.begin() != list2.end());
  }
  // Test iterators
  {
    decltype(list)::iterator it = list.begin();
    CHECK(it == list.begin());
    CHECK_FALSE(it != list.begin());
    decltype(list)::iterator it2 = it;
    CHECK(it2 == it);
    CHECK_FALSE(it2 != it);
    CHECK(*it == list(4.0));
    CHECK(it->update == list(4.0).update);
    {
      const auto& ret = ++it;
      CHECK(&ret == &it);
    }
    CHECK(it != it2);
    CHECK_FALSE(it == it2);
    CHECK(*it == list(2.0));
    {
      const auto old_it2 = it2;
      CHECK(it2++ == old_it2);
    }
    CHECK(it == it2);
    CHECK_FALSE(it != it2);
    ++it;
    CHECK(it == list.end());
    CHECK_FALSE(it != list.end());
  }

  {
    ThreadsafeList<int> infinite_list(0.0);
    infinite_list.insert(0.0, 1, std::numeric_limits<double>::infinity());
    CHECK(infinite_list.expiration_after(
              std::numeric_limits<double>::infinity()) ==
          std::numeric_limits<double>::infinity());
    CHECK(infinite_list.expiration_after(1.0) ==
          std::numeric_limits<double>::infinity());
  }

  CHECK_THROWS_WITH(
      ([]() {
        ThreadsafeList<int> error(1.0);
        error.insert(2.0, 1, 3.0);
      }()),
      Catch::Matchers::ContainsSubstring("Tried to insert at time 2") and
          Catch::Matchers::ContainsSubstring(
              ", which is not the old expiration time 1"));
  CHECK_THROWS_WITH(
      ([]() {
        ThreadsafeList<int> error(1.0);
        error.insert(1.0, 1, 3.0);
        error.insert(4.0, 1, 3.0);
      }()),
      Catch::Matchers::ContainsSubstring("Tried to insert at time 4") and
          Catch::Matchers::ContainsSubstring(
              ", which is not the old expiration time 3"));
  CHECK_THROWS_WITH(
      ([]() {
        ThreadsafeList<int> error(2.0);
        error.insert(2.0, 1, 1.0);
      }()),
      Catch::Matchers::ContainsSubstring("Expiration time 1") and
          Catch::Matchers::ContainsSubstring(" is not after update time 2"));
  CHECK_THROWS_WITH(
      ([]() {
        ThreadsafeList<int> error(2.0);
        error.insert(2.0, 1, 2.0);
      }()),
      Catch::Matchers::ContainsSubstring("Expiration time 2") and
          Catch::Matchers::ContainsSubstring(" is not after update time 2"));
  CHECK_THROWS_WITH(
      ([]() {
        ThreadsafeList<int> error(2.0);
        // Probably get the same error without the insert, but
        // "Attempt to access an empty function of time." would also
        // be acceptable in that case.
        error.insert(2.0, 1, 3.0);
        error(1.0);
      }()),
      Catch::Matchers::ContainsSubstring("Requested time 1") and
          Catch::Matchers::ContainsSubstring(" precedes earliest time 2"));
  CHECK_THROWS_WITH(([]() { ThreadsafeList<int>(1.0)(1.0); }()),
                    Catch::Matchers::ContainsSubstring(
                        "Attempt to access an empty function of time."));
  CHECK_THROWS_WITH(
      ([]() {
        ThreadsafeList<int> error(1.0);
        error.insert(1.0, 1, 3.0);
        error(4.0);
      }()),
      Catch::Matchers::ContainsSubstring("Attempt to evaluate at time 4") and
          Catch::Matchers::ContainsSubstring(
              ", which is after the expiration time 3"));
  CHECK_THROWS_WITH(
      ([]() {
        ThreadsafeList<int> error(1.0);
        error.insert(1.0, 1, 3.0);
        error.expiration_after(3.0);
      }()),
      Catch::Matchers::ContainsSubstring("Attempt to evaluate at time 3") and
          Catch::Matchers::ContainsSubstring(
              ", which is after the expiration time 3"));
}
