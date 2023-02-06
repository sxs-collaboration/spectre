// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <tuple>

#include "Utilities/CartesianProduct.hpp"
#include "Utilities/MakeArray.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.CartesianProduct", "[Unit][Utilities]") {
  {
    const auto result =
        cartesian_product(make_array(true, false), make_array(1, 2, 3));
    const std::array<std::tuple<bool, int>, 6> expected{
        {{true, 1}, {true, 2}, {true, 3}, {false, 1}, {false, 2}, {false, 3}}};
    CHECK(result == expected);
  }
}
