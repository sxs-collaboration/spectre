// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"

/// \cond
class ComplexDataVector;
/// \endcond
namespace {
struct SomeTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, 0>>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Tags", "[Unit][Cce]") {
  TestHelpers::db::test_simple_tag<Cce::Tags::BondiBeta>("BondiBeta");
  TestHelpers::db::test_simple_tag<Cce::Tags::BondiH>("H");
  TestHelpers::db::test_simple_tag<Cce::Tags::BondiJ>("J");
  TestHelpers::db::test_simple_tag<Cce::Tags::BondiJbar>("Jbar");
  TestHelpers::db::test_simple_tag<Cce::Tags::BondiK>("K");
  TestHelpers::db::test_simple_tag<Cce::Tags::BondiQ>("Q");
  TestHelpers::db::test_simple_tag<Cce::Tags::BondiQbar>("Qbar");
  TestHelpers::db::test_simple_tag<Cce::Tags::BondiU>("U");
  TestHelpers::db::test_simple_tag<Cce::Tags::BondiUAtScri>("BondiUAtScri");
  TestHelpers::db::test_simple_tag<Cce::Tags::BondiUbar>("Ubar");
  TestHelpers::db::test_simple_tag<Cce::Tags::BondiW>("W");
  TestHelpers::db::test_simple_tag<Cce::Tags::GaugeC>("GaugeC");
  TestHelpers::db::test_simple_tag<Cce::Tags::GaugeD>("GaugeD");
  TestHelpers::db::test_simple_tag<Cce::Tags::GaugeOmega>("GaugeOmega");
  TestHelpers::db::test_simple_tag<Cce::Tags::News>("News");
  TestHelpers::db::test_simple_tag<Cce::Tags::CauchyAngularCoords>(
      "CauchyAngularCoords");
  TestHelpers::db::test_simple_tag<Cce::Tags::CauchyCartesianCoords>(
      "CauchyCartesianCoords");
  TestHelpers::db::test_simple_tag<Cce::Tags::InertialRetardedTime>(
      "InertialRetardedTime");
  TestHelpers::db::test_simple_tag<Cce::Tags::ComplexInertialRetardedTime>(
      "ComplexInertialRetardedTime");
  TestHelpers::db::test_simple_tag<Cce::Tags::OneMinusY>("OneMinusY");
  TestHelpers::db::test_simple_tag<Cce::Tags::DuR>("DuR");
  TestHelpers::db::test_simple_tag<Cce::Tags::DuRDividedByR>("DuRDividedByR");
  TestHelpers::db::test_simple_tag<Cce::Tags::EthRDividedByR>("EthRDividedByR");
  TestHelpers::db::test_simple_tag<Cce::Tags::EthEthRDividedByR>(
      "EthEthRDividedByR");
  TestHelpers::db::test_simple_tag<Cce::Tags::EthEthbarRDividedByR>(
      "EthEthbarRDividedByR");
  TestHelpers::db::test_simple_tag<Cce::Tags::Exp2Beta>("Exp2Beta");
  TestHelpers::db::test_simple_tag<Cce::Tags::JbarQMinus2EthBeta>(
      "JbarQMinus2EthBeta");
  TestHelpers::db::test_simple_tag<Cce::Tags::BondiR>("R");
  TestHelpers::db::test_simple_tag<Cce::Tags::H5WorldtubeBoundaryDataManager>(
      "H5WorldtubeBoundaryDataManager");

  auto box =
      db::create<db::AddSimpleTags<Cce::Tags::H5WorldtubeBoundaryDataManager>>(
          Cce::WorldtubeDataManager{});
  CHECK(db::get<Cce::Tags::H5WorldtubeBoundaryDataManager>(box).get_l_max() ==
        0);

  TestHelpers::db::test_simple_tag<Cce::Tags::Psi0>("Psi0");
  TestHelpers::db::test_simple_tag<Cce::Tags::Psi1>("Psi1");
  TestHelpers::db::test_simple_tag<Cce::Tags::Psi2>("Psi2");
  TestHelpers::db::test_simple_tag<Cce::Tags::Psi3>("Psi3");
  TestHelpers::db::test_simple_tag<Cce::Tags::Psi4>("Psi4");
  TestHelpers::db::test_simple_tag<Cce::Tags::Strain>("Strain");

  TestHelpers::db::test_prefix_tag<Cce::Tags::Dy<SomeTag>>("Dy(SomeTag)");
  TestHelpers::db::test_prefix_tag<Cce::Tags::Du<SomeTag>>("Du(SomeTag)");
  TestHelpers::db::test_prefix_tag<Cce::Tags::Dr<SomeTag>>("Dr(SomeTag)");
  TestHelpers::db::test_prefix_tag<Cce::Tags::Integrand<SomeTag>>(
      "Integrand(SomeTag)");
  TestHelpers::db::test_prefix_tag<Cce::Tags::BoundaryValue<SomeTag>>(
      "BoundaryValue(SomeTag)");
  TestHelpers::db::test_prefix_tag<
      Cce::Tags::EvolutionGaugeBoundaryValue<SomeTag>>(
      "EvolutionGaugeBoundaryValue(SomeTag)");
  TestHelpers::db::test_prefix_tag<Cce::Tags::PoleOfIntegrand<SomeTag>>(
      "PoleOfIntegrand(SomeTag)");
  TestHelpers::db::test_prefix_tag<Cce::Tags::RegularIntegrand<SomeTag>>(
      "RegularIntegrand(SomeTag)");
  TestHelpers::db::test_prefix_tag<Cce::Tags::LinearFactor<SomeTag>>(
      "LinearFactor(SomeTag)");
  TestHelpers::db::test_prefix_tag<
      Cce::Tags::LinearFactorForConjugate<SomeTag>>(
      "LinearFactorForConjugate(SomeTag)");
  TestHelpers::db::test_prefix_tag<Cce::Tags::TimeIntegral<SomeTag>>(
      "TimeIntegral(SomeTag)");
  TestHelpers::db::test_prefix_tag<Cce::Tags::ScriPlus<SomeTag>>(
      "ScriPlus(SomeTag)");
  TestHelpers::db::test_prefix_tag<Cce::Tags::ScriPlusFactor<SomeTag>>(
      "ScriPlusFactor(SomeTag)");
}
