// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace {
struct ArbitraryFrame;
}  // namespace

template <size_t Dim, typename Frame, typename DataType>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<Ccz4::Tags::ConformalFactor<DataType>>(
      "ConformalFactor");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::ConformalMetric<Dim, Frame, DataType>>(
      "Conformal(SpatialMetric)");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::InverseConformalMetric<Dim, Frame, DataType>>(
      "Conformal(InverseSpatialMetric)");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::LogLapse<DataType>>("LogLapse");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::FieldA<Dim, Frame, DataType>>(
      "FieldA");
  TestHelpers::db::test_simple_tag<
      Ccz4::Tags::FieldB<Dim, Frame, DataType>>("FieldB");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::FieldD<Dim, Frame, DataType>>(
      "FieldD");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::LogConformalFactor<DataType>>(
      "LogConformalFactor");
  TestHelpers::db::test_simple_tag<Ccz4::Tags::FieldP<Dim, Frame, DataType>>(
      "FieldP");
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Tags", "[Unit][Evolution]") {
  test_simple_tags<1, ArbitraryFrame, double>();
  test_simple_tags<1, ArbitraryFrame, DataVector>();
  test_simple_tags<2, ArbitraryFrame, double>();
  test_simple_tags<2, ArbitraryFrame, DataVector>();
  test_simple_tags<3, ArbitraryFrame, double>();
  test_simple_tags<3, ArbitraryFrame, DataVector>();
}
