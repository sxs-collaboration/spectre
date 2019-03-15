// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

struct DataVector;

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.GeneralRelativity.Tags",
                  "[Unit][PointwiseFunctions]") {
  CHECK(gr::Tags::EnergyDensity<DataVector>::name() == "EnergyDensity");
}
