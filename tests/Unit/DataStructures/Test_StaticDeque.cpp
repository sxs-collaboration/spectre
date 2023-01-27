// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <deque>
#include <initializer_list>
#include <ostream>
#include <pup.h>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include "DataStructures/StaticDeque.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace {
struct AllocationChecker {
  int value{12345};

  AllocationChecker() { register_self(); }
  AllocationChecker(const AllocationChecker& other) : value(other.value) {
    register_self();
  }
  AllocationChecker& operator=(const AllocationChecker&) = default;

  ~AllocationChecker() { unregister_self(); }

  AllocationChecker(const int v) : value(v) { register_self(); }

  static bool was_clean() {
    const bool clean = live_objects_.empty();
    live_objects_.clear();
    return clean;
  }

  void pup(PUP::er& p) { p | value; }

 private:
  void register_self() const {
    const bool inserted = live_objects_.insert(this).second;
    REQUIRE(inserted);
  }

  void unregister_self() const {
    const int removed = live_objects_.erase(this);
    REQUIRE(removed == 1);
  }

  static std::unordered_set<const AllocationChecker*> live_objects_;
};

std::unordered_set<const AllocationChecker*> AllocationChecker::live_objects_{};

bool operator==(const AllocationChecker& a, const AllocationChecker& b) {
  return a.value == b.value;
}
[[maybe_unused]] bool operator!=(const AllocationChecker& a,
                                 const AllocationChecker& b) {
  return not(a == b);
}

std::ostream& operator<<(std::ostream& s, const AllocationChecker& a) {
  return s << a.value;
}

template <typename F>
void compare_with_stl_impl(const F operation) {
  {
    std::deque<AllocationChecker> stl_deque{1, 2, 3, 4, 5};
    StaticDeque<AllocationChecker, 9> static_deque{1, 2, 3, 4, 5};
    // Get the internals into a more interesting state.
    stl_deque.push_front(0);
    static_deque.push_front(0);
    if constexpr (std::is_same_v<decltype(operation(make_not_null(&stl_deque))),
                                 void>) {
      operation(make_not_null(&stl_deque));
      operation(make_not_null(&static_deque));
    } else if constexpr (std::is_same_v<decltype(operation(
                                            make_not_null(&stl_deque))),
                                        decltype(operation(
                                            make_not_null(&static_deque)))>) {
      CHECK(operation(make_not_null(&stl_deque)) ==
            operation(make_not_null(&static_deque)));
    } else {
      // Probably returning iterators

      // These evaluations cannot be inlined because the value of
      // .begin() could depend on evaluation order.
      const auto stl_result = operation(make_not_null(&stl_deque));
      const auto static_result = operation(make_not_null(&static_deque));
      CHECK((stl_result - stl_deque.begin()) ==
            (static_result - static_deque.begin()));
    }
    CAPTURE(stl_deque);
    CAPTURE(static_deque);
    CHECK(stl_deque.size() == static_deque.size());
    CHECK(std::equal(stl_deque.begin(), stl_deque.end(), static_deque.begin(),
                     static_deque.end()));
  }
  CHECK(AllocationChecker::was_clean());
}

#define STRINGIFY_LINE(line) STRINGIFY_LINE2(line)
#define STRINGIFY_LINE2(line) #line

#define COMPARE_WITH_STL(operation)         \
  do {                                      \
    INFO("Line " STRINGIFY_LINE(__LINE__)); \
    INFO(#operation);                       \
    compare_with_stl_impl(operation);       \
  } while (false)

template <int>
struct ForwardingTesterArg {
  ForwardingTesterArg() = delete;
  ForwardingTesterArg(const ForwardingTesterArg&) = delete;
  ForwardingTesterArg(ForwardingTesterArg&&) { CHECK(false); }
  ForwardingTesterArg& operator=(const ForwardingTesterArg&) = delete;
  ForwardingTesterArg& operator=(ForwardingTesterArg&&) {
    CHECK(false);
    return *this;
  }

  explicit ForwardingTesterArg(int) {}
};

struct ForwardingTester {
  ForwardingTester() = delete;
  ForwardingTester(const ForwardingTester&) = delete;
  ForwardingTester(ForwardingTester&&) = default;
  ForwardingTester& operator=(const ForwardingTester&) = delete;
  ForwardingTester& operator=(ForwardingTester&&) {
    CHECK(false);
    return *this;
  }

  ForwardingTester(ForwardingTesterArg<0>&&, ForwardingTesterArg<1>&&) {}
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.StaticDeque", "[Unit][DataStructures]") {
  { CHECK(StaticDeque<int, 3>{}.size() == 0); }
  {
    const StaticDeque<int, 4> deque(3, 6);
    CHECK(deque.size() == 3);
    CHECK(deque[0] == 6);
    CHECK(deque[1] == 6);
    CHECK(deque[2] == 6);
  }
  {
    const StaticDeque<int, 4> deque(3);
    CHECK(deque.size() == 3);
    CHECK(deque[0] == 0);
    CHECK(deque[1] == 0);
    CHECK(deque[2] == 0);
  }
  {
    const StaticDeque<int, 4> deque{3, 4, 5};
    CHECK(deque.size() == 3);
    CHECK(deque[0] == 3);
    CHECK(deque[1] == 4);
    CHECK(deque[2] == 5);
  }
  {
    std::initializer_list<int> init{3, 4, 5};
    const StaticDeque<int, 4> deque(init.begin(), init.end());
    CHECK(deque.size() == 3);
    CHECK(deque[0] == 3);
    CHECK(deque[1] == 4);
    CHECK(deque[2] == 5);
  }
  {
    const StaticDeque<int, 4> deque1{3, 4, 5};
    const StaticDeque<int, 4> deque2 = deque1;
    CHECK(deque1.size() == 3);
    CHECK(deque1[0] == 3);
    CHECK(deque1[1] == 4);
    CHECK(deque1[2] == 5);
    CHECK(deque2.size() == 3);
    CHECK(deque2[0] == 3);
    CHECK(deque2[1] == 4);
    CHECK(deque2[2] == 5);
  }
  {
    StaticDeque<int, 4> deque1{3, 4, 5};
    const StaticDeque<int, 4> deque2 = std::move(deque1);
    CHECK(deque2.size() == 3);
    CHECK(deque2[0] == 3);
    CHECK(deque2[1] == 4);
    CHECK(deque2[2] == 5);
  }
  {
    StaticDeque<NonCopyable, 4> deque1{};
    const StaticDeque<NonCopyable, 4> deque2 = std::move(deque1);
  }

  {
    const StaticDeque<int, 4> deque1{3, 4, 5};
    StaticDeque<int, 4> deque2{6, 7, 8, 9};
    deque2 = deque1;
    CHECK(deque1.size() == 3);
    CHECK(deque1[0] == 3);
    CHECK(deque1[1] == 4);
    CHECK(deque1[2] == 5);
    CHECK(deque2.size() == 3);
    CHECK(deque2[0] == 3);
    CHECK(deque2[1] == 4);
    CHECK(deque2[2] == 5);
  }
  {
    StaticDeque<int, 4> deque1{3, 4, 5};
    StaticDeque<int, 4> deque2{6, 7, 8, 9};
    deque2 = std::move(deque1);
    CHECK(deque2.size() == 3);
    CHECK(deque2[0] == 3);
    CHECK(deque2[1] == 4);
    CHECK(deque2[2] == 5);
  }
  {
    StaticDeque<NonCopyable, 4> deque1{};
    StaticDeque<NonCopyable, 4> deque2{};
    deque2 = std::move(deque1);
  }
  {
    StaticDeque<int, 4> deque{6, 7, 8, 9};
    deque = {3, 4, 5};
    CHECK(deque.size() == 3);
    CHECK(deque[0] == 3);
    CHECK(deque[1] == 4);
    CHECK(deque[2] == 5);
  }

  COMPARE_WITH_STL(([](const auto deque) {
    // Some of the checks below pick numbers above and below this.
    REQUIRE(deque->size() == 6);
  }));

  COMPARE_WITH_STL(([](const auto deque) { return deque->assign(4, 9); }));
  COMPARE_WITH_STL(([](const auto deque) { return deque->assign(8, 9); }));
  COMPARE_WITH_STL(([](const auto deque) {
    std::initializer_list<AllocationChecker> init{3, 4, 5};
    return deque->assign(init.begin(), init.end());
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    std::initializer_list<AllocationChecker> init{3, 4, 5};
    return deque->assign(init);
  }));

  COMPARE_WITH_STL(([](const auto deque) { return deque->at(0); }));
  COMPARE_WITH_STL(([](const auto deque) { return deque->at(5); }));
  CHECK_THROWS_AS(([]() {
                    StaticDeque<int, 4> deque{1, 2};
                    deque.at(2);
                  })(),
                  std::out_of_range);
  COMPARE_WITH_STL(
      ([](const auto deque) { return std::as_const(*deque).at(0); }));
  COMPARE_WITH_STL(
      ([](const auto deque) { return std::as_const(*deque).at(5); }));
  COMPARE_WITH_STL(([](const auto deque) { return (*deque)[0]; }));
  COMPARE_WITH_STL(([](const auto deque) { return (*deque)[5]; }));
  COMPARE_WITH_STL(([](const auto deque) { return std::as_const(*deque)[0]; }));
  COMPARE_WITH_STL(([](const auto deque) { return std::as_const(*deque)[5]; }));
  COMPARE_WITH_STL(([](const auto deque) { return deque->front(); }));
  COMPARE_WITH_STL(
      ([](const auto deque) { return std::as_const(*deque).front(); }));
  COMPARE_WITH_STL(([](const auto deque) { return deque->back(); }));
  COMPARE_WITH_STL(
      ([](const auto deque) { return std::as_const(*deque).back(); }));

  COMPARE_WITH_STL(([](const auto deque) { return deque->begin(); }));
  COMPARE_WITH_STL(
      ([](const auto deque) { return std::as_const(*deque).begin(); }));
  COMPARE_WITH_STL(
      ([](const auto deque) { return std::as_const(*deque).cbegin(); }));
  COMPARE_WITH_STL(([](const auto deque) { return deque->end(); }));
  COMPARE_WITH_STL(
      ([](const auto deque) { return std::as_const(*deque).end(); }));
  COMPARE_WITH_STL(
      ([](const auto deque) { return std::as_const(*deque).cend(); }));
  // Reverse iterators won't work with the macro.  This is all
  // inherited code tested elsewhere, anyway.

  {
    const StaticDeque<int, 3> deque{1};
    CHECK(not deque.empty());
  }
  {
    const StaticDeque<int, 3> deque{};
    CHECK(deque.empty());
  }
  COMPARE_WITH_STL(([](const auto deque) { return deque->size(); }));
  {
    const StaticDeque<int, 3> deque{};
    CHECK(deque.max_size() == 3);
  }
  COMPARE_WITH_STL(([](const auto deque) { return deque->shrink_to_fit(); }));

  COMPARE_WITH_STL(([](const auto deque) { return deque->clear(); }));

  // Test at each end and in each half.
  COMPARE_WITH_STL(([](const auto deque) {
    const AllocationChecker value = 9;
    return deque->insert(deque->begin(), value);
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    const AllocationChecker value = 9;
    return deque->insert(deque->end(), value);
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    const AllocationChecker value = 9;
    return deque->insert(deque->begin() + 2, value);
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    const AllocationChecker value = 9;
    return deque->insert(deque->end() - 2, value);
  }));

  COMPARE_WITH_STL(([](const auto deque) {
    AllocationChecker value = 9;
    return deque->insert(deque->begin(), std::move(value));
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    AllocationChecker value = 9;
    return deque->insert(deque->end(), std::move(value));
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    AllocationChecker value = 9;
    return deque->insert(deque->begin() + 2, std::move(value));
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    AllocationChecker value = 9;
    return deque->insert(deque->end() - 2, std::move(value));
  }));

  COMPARE_WITH_STL(
      ([](const auto deque) { return deque->insert(deque->begin(), 2, 9); }));
  COMPARE_WITH_STL(
      ([](const auto deque) { return deque->insert(deque->end(), 2, 9); }));
  COMPARE_WITH_STL(([](const auto deque) {
    return deque->insert(deque->begin() + 2, 2, 9);
  }));
  COMPARE_WITH_STL(
      ([](const auto deque) { return deque->insert(deque->end() - 2, 2, 9); }));

  COMPARE_WITH_STL(([](const auto deque) {
    std::initializer_list<AllocationChecker> init{3, 4, 5};
    return deque->insert(deque->begin(), init.begin(), init.end());
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    std::initializer_list<AllocationChecker> init{3, 4, 5};
    return deque->insert(deque->end(), init.begin(), init.end());
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    std::initializer_list<AllocationChecker> init{3, 4, 5};
    return deque->insert(deque->begin() + 2, init.begin(), init.end());
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    std::initializer_list<AllocationChecker> init{3, 4, 5};
    return deque->insert(deque->end() - 2, init.begin(), init.end());
  }));

  COMPARE_WITH_STL(([](const auto deque) {
    std::initializer_list<AllocationChecker> init{3, 4, 5};
    return deque->insert(deque->begin(), init);
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    std::initializer_list<AllocationChecker> init{3, 4, 5};
    return deque->insert(deque->end(), init);
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    std::initializer_list<AllocationChecker> init{3, 4, 5};
    return deque->insert(deque->begin() + 2, init);
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    std::initializer_list<AllocationChecker> init{3, 4, 5};
    return deque->insert(deque->end() - 2, init);
  }));

  COMPARE_WITH_STL(
      ([](const auto deque) { return deque->emplace(deque->begin(), 9); }));
  COMPARE_WITH_STL(
      ([](const auto deque) { return deque->emplace(deque->end(), 9); }));
  COMPARE_WITH_STL(
      ([](const auto deque) { return deque->emplace(deque->begin() + 2, 9); }));
  COMPARE_WITH_STL(
      ([](const auto deque) { return deque->emplace(deque->end() - 2, 9); }));

  COMPARE_WITH_STL(
      ([](const auto deque) { return deque->erase(deque->begin()); }));
  COMPARE_WITH_STL(
      ([](const auto deque) { return deque->erase(deque->end() - 1); }));
  COMPARE_WITH_STL(
      ([](const auto deque) { return deque->erase(deque->begin() + 2); }));
  COMPARE_WITH_STL(
      ([](const auto deque) { return deque->erase(deque->end() - 3); }));

  COMPARE_WITH_STL(([](const auto deque) {
    return deque->erase(deque->begin(), deque->begin() + 2);
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    return deque->erase(deque->end() - 3, deque->end() - 1);
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    return deque->erase(deque->begin() + 2, deque->begin() + 4);
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    return deque->erase(deque->end() - 5, deque->end() - 3);
  }));

  COMPARE_WITH_STL(([](const auto deque) {
    return deque->erase(deque->begin(), deque->begin());
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    return deque->erase(deque->end() - 1, deque->end() - 1);
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    return deque->erase(deque->begin() + 2, deque->begin() + 2);
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    return deque->erase(deque->end() - 3, deque->end() - 3);
  }));

  COMPARE_WITH_STL(([](const auto deque) {
    const AllocationChecker value = 9;
    return deque->push_back(value);
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    AllocationChecker value = 9;
    return deque->push_back(std::move(value));
  }));
  COMPARE_WITH_STL(([](const auto deque) { return deque->emplace_back(9); }));
  COMPARE_WITH_STL(([](const auto deque) { return deque->pop_back(); }));
  COMPARE_WITH_STL(([](const auto deque) {
    const AllocationChecker value = 9;
    return deque->push_front(value);
  }));
  COMPARE_WITH_STL(([](const auto deque) {
    AllocationChecker value = 9;
    return deque->push_front(std::move(value));
  }));
  COMPARE_WITH_STL(([](const auto deque) { return deque->emplace_front(9); }));
  COMPARE_WITH_STL(([](const auto deque) { return deque->pop_front(); }));

  COMPARE_WITH_STL(([](const auto deque) { return deque->resize(4); }));
  COMPARE_WITH_STL(([](const auto deque) { return deque->resize(6); }));
  COMPARE_WITH_STL(([](const auto deque) { return deque->resize(4, 9); }));
  COMPARE_WITH_STL(([](const auto deque) { return deque->resize(6, 9); }));

  {
    StaticDeque<int, 9> deque1{1, 2};
    StaticDeque<int, 9> deque2{2};
    // Different internal state
    deque2.push_front(1);
    CHECK(deque1 == deque2);
    CHECK_FALSE(deque1 != deque2);
    CHECK_FALSE(deque1 < deque2);
    CHECK_FALSE(deque1 > deque2);
    CHECK(deque1 <= deque2);
    CHECK(deque1 >= deque2);
  }
  {
    StaticDeque<int, 9> deque1{1, 2, 3};
    StaticDeque<int, 9> deque2{3};
    // Different internal state
    deque2.push_front(1);
    CHECK_FALSE(deque1 == deque2);
    CHECK(deque1 != deque2);
    CHECK(deque1 < deque2);
    CHECK_FALSE(deque1 > deque2);
    CHECK(deque1 <= deque2);
    CHECK_FALSE(deque1 >= deque2);
  }
  {
    StaticDeque<int, 9> deque1{1, 2, 3};
    StaticDeque<int, 9> deque2{2};
    // Different internal state
    deque2.push_front(1);
    CHECK_FALSE(deque1 == deque2);
    CHECK(deque1 != deque2);
    CHECK_FALSE(deque1 < deque2);
    CHECK(deque1 > deque2);
    CHECK_FALSE(deque1 <= deque2);
    CHECK(deque1 >= deque2);
  }

  {
    StaticDeque<AllocationChecker, 9> deque1{1, 2, 3, 4, 5};
    StaticDeque<AllocationChecker, 9> deque2{7, 8};
    deque1.swap(deque2);
    CHECK(deque1 == StaticDeque<AllocationChecker, 9>{7, 8});
    CHECK(deque2 == StaticDeque<AllocationChecker, 9>{1, 2, 3, 4, 5});
  }
  CHECK(AllocationChecker::was_clean());

  {
    StaticDeque<AllocationChecker, 9> deque1{1, 2, 3, 4, 5};
    StaticDeque<AllocationChecker, 9> deque2{7, 8};
    using std::swap;
    swap(deque1, deque2);
    CHECK(deque1 == StaticDeque<AllocationChecker, 9>{7, 8});
    CHECK(deque2 == StaticDeque<AllocationChecker, 9>{1, 2, 3, 4, 5});
  }
  CHECK(AllocationChecker::was_clean());

  // Check argument forwarding
  {
    StaticDeque<ForwardingTester, 10> deque{};
    deque.insert(deque.begin(), ForwardingTester(ForwardingTesterArg<0>(1),
                                                 ForwardingTesterArg<1>(1)));
    deque.emplace(deque.begin(), ForwardingTesterArg<0>(1),
                  ForwardingTesterArg<1>(1));
    deque.emplace(deque.end(), ForwardingTesterArg<0>(1),
                  ForwardingTesterArg<1>(1));
    deque.push_back(
        ForwardingTester(ForwardingTesterArg<0>(1), ForwardingTesterArg<1>(1)));
    deque.emplace_back(ForwardingTesterArg<0>(1), ForwardingTesterArg<1>(1));
    deque.push_front(
        ForwardingTester(ForwardingTesterArg<0>(1), ForwardingTesterArg<1>(1)));
    deque.emplace_front(ForwardingTesterArg<0>(1), ForwardingTesterArg<1>(1));
  }

  {
    StaticDeque<AllocationChecker, 7> deque{3, 4, 5};
    test_serialization(deque);
  }
  CHECK(AllocationChecker::was_clean());
}
