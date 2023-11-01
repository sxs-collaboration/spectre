// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "Utilities/SplitTuple.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// Test constexpr
// [split_tuple]
static_assert(split_tuple<2, 1>(std::tuple{1, 2, 3}) ==
              std::tuple{std::tuple{1, 2}, std::tuple{3}});
static_assert(split_tuple<tmpl::integral_list<size_t, 2, 1>>(std::tuple{
                  1, 2, 3}) == std::tuple{std::tuple{1, 2}, std::tuple{3}});
// [split_tuple]

// Needed for type-deduction of the nested tuples.
template <typename... T>
std::tuple<T...> construct_tuple(T... args) {
  return {std::move(args)...};
}

template <size_t... Sizes, typename... T, typename Expected>
void check(std::tuple<T...> tuple, const Expected& expected) {
  auto split = split_tuple<Sizes...>(tuple);
  static_assert(std::is_same_v<decltype(split), Expected>);
  auto split_typelist =
      split_tuple<tmpl::integral_list<size_t, Sizes...>>(tuple);
  static_assert(std::is_same_v<decltype(split_typelist), Expected>);
  CHECK(split == expected);
  CHECK(split_typelist == expected);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.SplitTuple", "[Unit][Utilities]") {
  // Normal case
  check<2, 1>(construct_tuple<int, double, float>(1, 2.0, 3.0),
              construct_tuple<std::tuple<int, double>, std::tuple<float>>(
                  {1, 2.0}, {3.0}));
  // Empty piece
  check<2, 0, 1>(
      construct_tuple<int, double, float>(1, 2.0, 3.0),
      construct_tuple<std::tuple<int, double>, std::tuple<>, std::tuple<float>>(
          {1, 2.0}, {}, {3.0}));
  // No output
  check<>(construct_tuple<>(), construct_tuple<>());
  // Single output
  check<2>(construct_tuple<int, double>(1, 2.0),
           construct_tuple<std::tuple<int, double>>({1, 2.0}));
  // Non-copyable
  {
    using Expected =
        std::tuple<std::tuple<int>, std::tuple<std::unique_ptr<double>>>;
    auto split =
        split_tuple<1, 1>(construct_tuple<int, std::unique_ptr<double>>(
            1, std::make_unique<double>(2.0)));
    static_assert(std::is_same_v<decltype(split), Expected>);
    auto split_typelist = split_tuple<tmpl::integral_list<size_t, 1, 1>>(
        construct_tuple<int, std::unique_ptr<double>>(
            1, std::make_unique<double>(2.0)));
    static_assert(std::is_same_v<decltype(split_typelist), Expected>);
    CHECK(std::get<0>(std::get<0>(split)) == 1);
    CHECK(*std::get<0>(std::get<1>(split)) == 2.0);
    CHECK(std::get<0>(std::get<0>(split_typelist)) == 1);
    CHECK(*std::get<0>(std::get<1>(split_typelist)) == 2.0);
  }
  // References
  {
    // NOLINTBEGIN(bugprone-use-after-move)
    const int a = 0;
    double b = 0.0;
    using Expected = std::tuple<std::tuple<const int&>, std::tuple<double&&>>;
    auto split = split_tuple<1, 1>(
        construct_tuple<const int&, double&&>(a, std::move(b)));
    static_assert(std::is_same_v<decltype(split), Expected>);
    auto split_typelist = split_tuple<tmpl::integral_list<size_t, 1, 1>>(
        construct_tuple<const int&, double&&>(a, std::move(b)));
    static_assert(std::is_same_v<decltype(split_typelist), Expected>);
    CHECK(&std::get<0>(std::get<0>(split)) == &a);
    CHECK(&std::get<0>(std::get<1>(split)) == &b);
    CHECK(&std::get<0>(std::get<0>(split_typelist)) == &a);
    CHECK(&std::get<0>(std::get<1>(split_typelist)) == &b);
    // NOLINTEND(bugprone-use-after-move)
  }
}
