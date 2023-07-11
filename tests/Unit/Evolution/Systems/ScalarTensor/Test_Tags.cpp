// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct ArbitraryFrame;
}  // namespace

template <size_t Dim, typename Frame>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<
      ScalarTensor::Tags::TraceReversedStressEnergy<DataVector, Dim, Frame>>(
      "TraceReversedStressEnergy");
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarTensor.Tags",
                  "[Unit][Evolution]") {
  test_simple_tags<1_st, ArbitraryFrame>();
  test_simple_tags<2_st, ArbitraryFrame>();
  test_simple_tags<3_st, ArbitraryFrame>();
}
