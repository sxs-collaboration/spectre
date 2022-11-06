// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Evolution/Systems/GeneralizedHarmonic/DuDtTempTags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

template <size_t Dim>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<GeneralizedHarmonic::Tags::Gamma1Gamma2>(
      "Gamma1Gamma2");
  TestHelpers::db::test_simple_tag<GeneralizedHarmonic::Tags::PiTwoNormals>(
      "PiTwoNormals");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::NormalDotOneIndexConstraint>(
      "NormalDotOneIndexConstraint");
  TestHelpers::db::test_simple_tag<GeneralizedHarmonic::Tags::Gamma1Plus1>(
      "Gamma1Plus1");
  TestHelpers::db::test_simple_tag<GeneralizedHarmonic::Tags::PiOneNormal<Dim>>(
      "PiOneNormal");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::PhiTwoNormals<Dim>>("PhiTwoNormals");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::ShiftDotThreeIndexConstraint<Dim>>(
      "ShiftDotThreeIndexConstraint");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::MeshVelocityDotThreeIndexConstraint<Dim>>(
      "MeshVelocityDotThreeIndexConstraint");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::PhiOneNormal<Dim>>("PhiOneNormal");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::PiSecondIndexUp<Dim>>("PiSecondIndexUp");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::PhiFirstIndexUp<Dim>>("PhiFirstIndexUp");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::PhiThirdIndexUp<Dim>>("PhiThirdIndexUp");
  TestHelpers::db::test_simple_tag<
      GeneralizedHarmonic::Tags::SpacetimeChristoffelFirstKindThirdIndexUp<
          Dim>>("SpacetimeChristoffelFirstKindThirdIndexUp");
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.DuDtTempTags",
                  "[Unit][Evolution]") {
  test_simple_tags<1>();
  test_simple_tags<2>();
  test_simple_tags<3>();
}
