// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ContainerHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.ContainerHelpers", "[Unit][Utilities]") {
  const std::vector<double> a{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  const double b = 10.0;
  const DataVector c(10, 2.0);
  std::vector<double> a2{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  double b2 = 10.0;
  DataVector c2(10, 2.0);
  REQUIRE(get_size(a) == 10);
  REQUIRE(get_size(b) == 1);
  REQUIRE(get_size(c) == 10);
  REQUIRE(get_size(a2) == 10);
  REQUIRE(get_size(b2) == 1);
  REQUIRE(get_size(c2) == 10);
  for (size_t i = 0; i < a.size(); ++i) {
    get_element(a2, i) = get_element(a, i) * 2.0;
    get_element(b2, i) = get_element(b, i) * 2.0;
    get_element(c2, i) = get_element(c, i) * 2.0;
    CHECK(get_element(a, i) == static_cast<double>(i));
    CHECK(get_element(b, i) == 10.0);
    CHECK(get_element(c, i) == 2.0);
    CHECK(get_element(a2, i) == 2.0 * static_cast<double>(i));
    CHECK(get_element(b2, i) == 20.0);
    CHECK(get_element(c2, i) == 4.0);
  }
}
