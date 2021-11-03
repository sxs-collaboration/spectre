// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/Ccz4/TempTags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct ArbitraryFrame;
}  // namespace

template <size_t Dim, typename Frame, typename DataType>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::GammaHatMinusContractedConformalChristoffel<Dim, Frame,
                                                              DataType>>(
      "GammaHatMinusContractedConformalChristoffel");
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.TempTags", "[Unit][Evolution]") {
  test_simple_tags<1, ArbitraryFrame, double>();
  test_simple_tags<1, ArbitraryFrame, DataVector>();
  test_simple_tags<2, ArbitraryFrame, double>();
  test_simple_tags<2, ArbitraryFrame, DataVector>();
  test_simple_tags<3, ArbitraryFrame, double>();
  test_simple_tags<3, ArbitraryFrame, DataVector>();
}
