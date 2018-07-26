// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <vector>

#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
template <typename T>
struct MyNonCopyable {
  MyNonCopyable() = default;
  ~MyNonCopyable() = default;
  explicit MyNonCopyable(T t) : t_(std::move(t)) {}
  MyNonCopyable(const MyNonCopyable&) = delete;
  MyNonCopyable& operator=(const MyNonCopyable&) = delete;
  MyNonCopyable(MyNonCopyable&&) = default;
  MyNonCopyable& operator=(MyNonCopyable&&) = default;

  const T& get() const noexcept { return t_; }

 private:
  T t_;
};
}  // namespace

static_assert(noexcept(make_array<2>(0)),
              "Failed Unit.Utilities.MakeArray testing noexcept calculation of "
              "make_array.");
static_assert(noexcept(make_array<2>(DoesNotThrow{})),
              "Failed Unit.Utilities.MakeArray testing noexcept calculation of "
              "make_array.");
static_assert(not noexcept(make_array<2>(DoesThrow{})),
              "Failed Unit.Utilities.MakeArray testing noexcept calculation of "
              "make_array.");

SPECTRE_TEST_CASE("Unit.Utilities.MakeArray", "[Unit][Utilities]") {
  constexpr auto same_array = make_array<4>(7.8);
  CHECK((same_array == std::array<double, 4>{{7.8, 7.8, 7.8, 7.8}}));
  constexpr auto varying_array = make_array(3.2, 4.3, 5.4, 6.5, 7.8);
  CHECK((varying_array == std::array<double, 5>{{3.2, 4.3, 5.4, 6.5, 7.8}}));

  auto array_n = make_array<6>(5);
  static_assert(
      std::is_same<typename std::decay<decltype(array_n)>::type::value_type,
                   int>::value,
      "Unit Test Failure: Incorrect type from make_array.");
  static_assert(array_n.size() == 6,
                "Unit Test Failure: Incorrect size from make_array.");
  for (const auto& p : array_n) {
    CHECK(5 == p);
  }

  auto array_non_copyable = make_array<1>(NonCopyable{});
  static_assert(array_non_copyable.size() == 1,
                "Unit Test Failure: Incorrect array size should be 1 for "
                "move-only types.");

  auto array_empty = make_array<0>(2);
  static_assert(array_empty.empty(),
                "Unit Test Failure: Incorrect array size for empty array.");

  auto array_non_copyable_empty = make_array<0>(NonCopyable{});
  static_assert(
      array_non_copyable_empty.empty(),
      "Unit Test Failure: Incorrect array size for empty array of move-only.");

  auto my_array = make_array(1, 3, 4, 8, 9);
  static_assert(my_array.size() == 5,
                "Unit Test Failure: Incorrect array size.");
  CHECK((my_array[0] == 1 and my_array[1] == 3 and my_array[2] == 4 and
         my_array[3] == 8 and my_array[4] == 9));

  constexpr const auto array_from_sequence =
      make_array<int, 3>(std::initializer_list<int>{2, 8, 6});
  CHECK((std::array<int, 3>{{2, 8, 6}}) == array_from_sequence);
  constexpr const auto array_from_truncated_sequence =
      make_array<int, 3>(std::initializer_list<int>{2, 8, 6, 9, 7});
  CHECK((std::array<int, 3>{{2, 8, 6}}) == array_from_truncated_sequence);

  // Check make_array with a non-copyable type
  const auto noncopyable_vectors =
      make_array<3, MyNonCopyable<std::vector<double>>>(
          std::vector<double>{2.3, 8.6, 9.7});
  for (size_t i = 0; i < 3; i++) {
    CHECK(gsl::at(noncopyable_vectors, i).get() ==
          (std::vector<double>{2.3, 8.6, 9.7}));
  }

  // Check make_array with a multi-arg constructor
  const auto vector_array1 = make_array<3, std::vector<int>>(3_st, 9);
  for (size_t i = 0; i < 3; i++) {
    CHECK(gsl::at(vector_array1, i) == (std::vector<int>(3, 9)));
  }

  // Check make_array with an initializer list constructor
  const auto vector_array2 = make_array<3, std::vector<int>>({3, 9, 12, -10});
  for (size_t i = 0; i < 3; i++) {
    CHECK(gsl::at(vector_array2, i) == (std::vector<int>{3, 9, 12, -10}));
  }

  // Check make_array with an rvalue sequence
  {
    std::vector<MyNonCopyable<int>> vector;
    vector.reserve(3);
    for (int i = 0; i < 3; ++i) {
      vector.emplace_back(i);
    }
    const auto array = make_array<MyNonCopyable<int>, 3>(std::move(vector));
    for (size_t i = 0; i < 3; ++i) {
      CHECK(gsl::at(array, i).get() == i);
    }
  }

  // Check that make_array does not move from an lvalue sequence
  {
    std::vector<std::vector<int>> vector(3, std::vector<int>{1, 2, 3});
    make_array<std::vector<int>, 3>(vector);
    CHECK(vector.size() == 3);
    for (size_t i = 0; i < 3; ++i) {
      CHECK(vector[i] == (std::vector<int>{1, 2, 3}));
    }
  }

  // Check that make_array works with non-default-constructible types
  {
    struct NonDefaultConstructible {
      NonDefaultConstructible() = delete;
      explicit NonDefaultConstructible(int /*unused*/) noexcept {}
    };
    const NonDefaultConstructible ndc{1};
    // We just check that these compile, since there's not any
    // realistic way they could give a wrong answer.
    make_array<0>(ndc);
    make_array<1>(ndc);
    make_array<2>(ndc);
    make_array(ndc, ndc);
    make_array(ndc, ndc, ndc);
  }
}
