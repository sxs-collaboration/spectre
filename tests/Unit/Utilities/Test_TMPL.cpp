// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <tuple>
#include <utility>

#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace {
/// [expand_pack_example]
template <typename... Elements, size_t... Is>
void transform(const std::tuple<Elements...>& tupull,
               std::tuple<Elements...>& out_tupull,
               std::index_sequence<Is...> /*meta*/) {
  const auto func = [](const auto& in, auto& out) {
    out = in * static_cast<decltype(in)>(2);
    return 0;
  };
  expand_pack(func(std::get<Is>(tupull), std::get<Is>(out_tupull))...);
}

void test_expand_pack() noexcept {
  std::tuple<int, double, float> my_tupull = std::make_tuple(3, 2.7, 8.2);
  std::tuple<int, double, float> my_tupull_output;
  transform(my_tupull, my_tupull_output, std::make_index_sequence<3>{});
  CHECK(std::get<0>(my_tupull_output) == 6);
  CHECK(std::get<1>(my_tupull_output) == 5.4);
  CHECK(std::get<2>(my_tupull_output) == 16.4f);
}
/// [expand_pack_example]

template <typename>
struct Templated {};

static_assert(tmpl::list_contains_v<tmpl::list<Templated<int>,
                                               Templated<double>>,
                                    Templated<double>>,
              "Failed testing list_contains");

static_assert(not tmpl::list_contains_v<tmpl::list<Templated<int>,
                                                   Templated<double>>,
                                        double>,
              "Failed testing list_contains");

static_assert(
    std::is_same_v<
        tmpl::list_difference<tmpl::list<Templated<int>, Templated<double>>,
                              tmpl::list<double>>,
        tmpl::list<Templated<int>, Templated<double>>>,
    "Failed testing list_difference");

static_assert(
    std::is_same_v<
        tmpl::list_difference<tmpl::list<Templated<int>, Templated<double>>,
                              tmpl::list<Templated<double>>>,
        tmpl::list<Templated<int>>>,
    "Failed testing list_difference");

static_assert(
    std::is_same_v<
        tmpl::list_difference<tmpl::list<>, tmpl::list<Templated<double>>>,
        tmpl::list<>>,
    "Failed testing list_difference");

void test_get_first_argument() noexcept {
  const long a0 = 5;
  const long a1 = 6;
  const int a2 = -5;
  const char a3 = '7';
  CHECK(5 == get_first_argument(a0, a1, a2, a3));
  CHECK(5 == get_first_argument(a0));
  CHECK(6 == get_first_argument(a1, a0, a2, a3));
  CHECK('7' == get_first_argument(a3, a1, a2, a0));
}

/// [expand_pack_left_to_right]
template <typename... Ts>
void test_expand_pack_left_to_right(const size_t expected,
                                    tmpl::list<Ts...> /*meta*/) {
  size_t sum = 0;
  const auto lambda = [&sum](auto tag) { sum += decltype(tag)::value; };
  EXPAND_PACK_LEFT_TO_RIGHT(lambda(Ts{}));
  CHECK(sum == expected);
}
/// [expand_pack_left_to_right]

static_assert(tmpl::as_pack<tmpl::list<tmpl::size_t<1>, tmpl::size_t<2>,
                                       tmpl::size_t<3>>>([](auto... args) {
                return (... + tmpl::type_from<decltype(args)>::value);
              }) == 6);

void test_as_pack() noexcept {
  /// [as_pack]
  using List = tmpl::list<tmpl::size_t<1>, tmpl::size_t<2>, tmpl::size_t<3>>;
  const size_t result = tmpl::as_pack<List>([](auto... args) {
    return (... + tmpl::type_from<decltype(args)>::value);
  });
  CHECK(result == 6);
  /// [as_pack]
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.TMPL", "[Unit][Utilities]") {
  test_expand_pack();
  test_get_first_argument();
  test_expand_pack_left_to_right(
      10, tmpl::list<std::integral_constant<size_t, 2>,
                     std::integral_constant<size_t, 4>,
                     std::integral_constant<size_t, 1>,
                     std::integral_constant<size_t, 3>>{});
  test_as_pack();
}
