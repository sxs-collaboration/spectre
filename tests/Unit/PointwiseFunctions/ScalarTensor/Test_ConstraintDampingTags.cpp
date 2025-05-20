// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/ScalarTensor/ConstraintDampingTags.hpp"

namespace {
struct ArbitraryFrame;

template <size_t Dim, typename Fr>
void test_tags() {
  TestHelpers::db::test_simple_tag<
      ScalarTensor::Tags::DampingFunctionGamma1<Dim, Fr>>(
      "DampingFunctionGamma1");
  TestHelpers::db::test_simple_tag<
      ScalarTensor::Tags::DampingFunctionGamma2<Dim, Fr>>(
      "DampingFunctionGamma2");
}

template <size_t Dim, typename Fr>
void test_option_tags() {
  TestHelpers::test_option_tag<
      ScalarTensor::OptionTags::DampingFunctionGamma1<Dim, Fr>>(
      "Constant:\n"
      "  Value: 5.0\n");
  TestHelpers::test_option_tag<
      ScalarTensor::OptionTags::DampingFunctionGamma2<Dim, Fr>>(
      "Constant:\n"
      "  Value: 5.0\n");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.ScalarTensor.ConstraintDampTags",
                  "[Unit][PointwiseFunctions]") {
  test_tags<1, ArbitraryFrame>();
  test_tags<2, ArbitraryFrame>();
  test_tags<3, ArbitraryFrame>();
  test_option_tags<1, Frame::Grid>();
  test_option_tags<2, Frame::Grid>();
  test_option_tags<3, Frame::Grid>();
  test_option_tags<1, Frame::Inertial>();
  test_option_tags<2, Frame::Inertial>();
  test_option_tags<3, Frame::Inertial>();
}
