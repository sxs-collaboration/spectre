// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <list>
#include <ostream>
#include <sstream>
#include <utility>
#include <vector>

#include "Utilities/PrintHelpers.hpp"

namespace {
template <typename F, typename Arg>
void check_value_categories(const F& test, const Arg& arg) {
  auto copy = arg;
  test(std::as_const(copy));
  test(copy);
  REQUIRE(copy == arg);
  test(std::move(copy));
}

template <typename F, typename It>
void check_all_categories(const F& test, const It& begin, const It& end) {
  check_value_categories(
      [&end, &test](auto&& begin_cat) {
        check_value_categories(
            [&begin_cat, &test](auto&& end_cat) {
              // Need a copy for each call as it may be moved out of.
              auto begin_copy = begin_cat;
              // Having begin_cat not match begin_copy is intentional.
              test(std::forward<decltype(begin_cat)>(begin_copy),
                   std::forward<decltype(end_cat)>(end_cat));
            },
            end);
      },
      begin);
}

template <typename F>
void test_vector_and_list(const F& test) {
  const std::vector<int> vec{1, 2, 9, 8, 11};
  const std::list<int> list(vec.begin(), vec.end());

  // Pointers are valid iterators for contiguous containers.
  const int* const vec_begin = vec.data();
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  const int* const vec_end = vec.data() + vec.size();
  // std::list is guaranteed to have a class as an iterator.
  const auto list_begin = list.begin();
  const auto list_end = list.end();

  check_all_categories(test, vec_begin, vec_end);
  check_all_categories(test, list_begin, list_end);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.PrintHelpers", "[Utilities][Unit]") {
  const auto stars = [](std::ostream& os, const auto& it) {
    os << "*" << *it << "*";
  };

  test_vector_and_list([](auto&& begin, auto&& end) {
    std::ostringstream os{};
    sequence_print_helper(os, std::forward<decltype(begin)>(begin),
                          std::forward<decltype(end)>(end));
    CHECK(os.str() == "(1,2,9,8,11)");
  });

  test_vector_and_list([&stars](auto&& begin, auto&& end) {
    std::ostringstream os{};
    sequence_print_helper(os, std::forward<decltype(begin)>(begin),
                          std::forward<decltype(end)>(end), stars);
    CHECK(os.str() == "(*1*,*2*,*9*,*8*,*11*)");
  });

  test_vector_and_list([](auto&& begin, auto&& end) {
    std::ostringstream os{};
    unordered_print_helper(os, std::forward<decltype(begin)>(begin),
                           std::forward<decltype(end)>(end));
    CHECK(os.str() == "(1,11,2,8,9)");
  });

  test_vector_and_list([&stars](auto&& begin, auto&& end) {
    std::ostringstream os{};
    unordered_print_helper(os, std::forward<decltype(begin)>(begin),
                           std::forward<decltype(end)>(end), stars);
    CHECK(os.str() == "(*1*,*11*,*2*,*8*,*9*)");
  });
}
