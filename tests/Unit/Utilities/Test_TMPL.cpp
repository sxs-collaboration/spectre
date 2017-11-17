// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <tuple>
#include <utility>

#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"

/// [swallow_example]
namespace {
template <typename... Elements, size_t... Is>
void transform(const std::tuple<Elements...>& tupull,
               std::tuple<Elements...>& out_tupull,
               std::index_sequence<Is...> /*meta*/) {
  const auto func = [](const auto& in, auto& out) {
    out = in * static_cast<decltype(in)>(2);
    return 0;
  };
  swallow(func(std::get<Is>(tupull), std::get<Is>(out_tupull))...);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.swallow", "[Utilities][Unit]") {
  std::tuple<int, double, float> my_tupull = std::make_tuple(3, 2.7, 8.2);
  std::tuple<int, double, float> my_tupull_output;
  transform(my_tupull, my_tupull_output, std::make_index_sequence<3>{});
  CHECK(std::get<0>(my_tupull_output) == 6);
  CHECK(std::get<1>(my_tupull_output) == 5.4);
  CHECK(std::get<2>(my_tupull_output) == 16.4f);
}
/// [swallow_example]

namespace {
template <typename>
struct Templated {};
}  // namespace

static_assert(tmpl::list_contains_v<tmpl::list<Templated<int>,
                                               Templated<double>>,
                                    Templated<double>>,
              "Failed testing list_contains");

static_assert(not tmpl::list_contains_v<tmpl::list<Templated<int>,
                                                   Templated<double>>,
                                        double>,
              "Failed testing list_contains");
