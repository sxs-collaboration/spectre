// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingFunctions/Exponential.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingFunctions/QuarticPolynomial.hpp"
#include "PointwiseFunctions/ScalarTensor/ScalarGaussBonnet/CouplingFunctions/Tags.hpp"

SPECTRE_TEST_CASE(
  "Unit.PointwiseFunctions.ScalarTensor.sgb.CouplingFunctions.Tags",
  "[Unit][PointwiseFunctions]") {
  TestHelpers::db::test_simple_tag<ScalarTensor::sgb::Tags::Ell>("Ell");
  TestHelpers::db::test_simple_tag<ScalarTensor::sgb::Tags::CouplingFunction>(
      "CouplingFunction");

  TestHelpers::test_option_tag_factory_creation<
      ScalarTensor::sgb::OptionTags::CouplingFunction,
      ScalarTensor::sgb::CouplingFunctions::Exponential>(
      "Exponential:\n"
      "  lambda: 0.7\n"
      "  gamma: 4\n");
  TestHelpers::test_option_tag_factory_creation<
      ScalarTensor::sgb::OptionTags::CouplingFunction,
      ScalarTensor::sgb::CouplingFunctions::QuarticPolynomial>(
      "QuarticPolynomial:\n"
      "  Linear: 0.7\n"
      "  Quadratic: 4\n"
      "  Cubic: 21.6\n"
      "  Quartic: -5.7\n");
}
