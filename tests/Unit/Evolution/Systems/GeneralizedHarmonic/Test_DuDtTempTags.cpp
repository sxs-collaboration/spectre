// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/GeneralizedHarmonic/DuDtTempTags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

template <size_t Dim>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<gh::Tags::Gamma1Gamma2>("Gamma1Gamma2");
  TestHelpers::db::test_simple_tag<gh::Tags::HalfPiTwoNormals>(
      "HalfPiTwoNormals");
  TestHelpers::db::test_simple_tag<gh::Tags::NormalDotOneIndexConstraint>(
      "NormalDotOneIndexConstraint");
  TestHelpers::db::test_simple_tag<gh::Tags::Gamma1Plus1>("Gamma1Plus1");
  TestHelpers::db::test_simple_tag<gh::Tags::PiOneNormal<Dim>>("PiOneNormal");
  TestHelpers::db::test_simple_tag<gh::Tags::HalfPhiTwoNormals<Dim>>(
      "HalfPhiTwoNormals");
  TestHelpers::db::test_simple_tag<gh::Tags::ShiftDotThreeIndexConstraint<Dim>>(
      "ShiftDotThreeIndexConstraint");
  TestHelpers::db::test_simple_tag<
      gh::Tags::MeshVelocityDotThreeIndexConstraint<Dim>>(
      "MeshVelocityDotThreeIndexConstraint");
  TestHelpers::db::test_simple_tag<gh::Tags::PhiOneNormal<Dim>>("PhiOneNormal");
  TestHelpers::db::test_simple_tag<gh::Tags::PiSecondIndexUp<Dim>>(
      "PiSecondIndexUp");
  TestHelpers::db::test_simple_tag<gh::Tags::PhiFirstIndexUp<Dim>>(
      "PhiFirstIndexUp");
  TestHelpers::db::test_simple_tag<gh::Tags::PhiThirdIndexUp<Dim>>(
      "PhiThirdIndexUp");
  TestHelpers::db::test_simple_tag<
      gh::Tags::SpacetimeChristoffelFirstKindThirdIndexUp<Dim>>(
      "SpacetimeChristoffelFirstKindThirdIndexUp");
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.DuDtTempTags",
                  "[Unit][Evolution]") {
  test_simple_tags<1>();
  test_simple_tags<2>();
  test_simple_tags<3>();
}
