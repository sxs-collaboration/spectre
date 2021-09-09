// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/SizeOfElement.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodType.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/CharacteristicHelpers.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/Minmod.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

void test_neuler_minmod_option_parsing() noexcept {
  INFO("Testing option parsing");
  const auto lambda_pi1 =
      TestHelpers::test_creation<grmhd::ValenciaDivClean::Limiters::Minmod>(
          "Type: LambdaPi1\n"
          "VariablesToLimit: NumericalCharacteristic\n"
          "TvbConstant: 0.0\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto lambda_pi1_cons =
      TestHelpers::test_creation<grmhd::ValenciaDivClean::Limiters::Minmod>(
          "Type: LambdaPi1\n"
          "VariablesToLimit: Conserved\n"
          "TvbConstant: 0.0\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto lambda_pi1_tvb =
      TestHelpers::test_creation<grmhd::ValenciaDivClean::Limiters::Minmod>(
          "Type: LambdaPi1\n"
          "VariablesToLimit: NumericalCharacteristic\n"
          "TvbConstant: 1.0\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto lambda_pi1_noflat =
      TestHelpers::test_creation<grmhd::ValenciaDivClean::Limiters::Minmod>(
          "Type: LambdaPi1\n"
          "VariablesToLimit: NumericalCharacteristic\n"
          "TvbConstant: 0.0\n"
          "ApplyFlattener: False\n"
          "DisableForDebugging: False");
  const auto lambda_pi1_disabled =
      TestHelpers::test_creation<grmhd::ValenciaDivClean::Limiters::Minmod>(
          "Type: LambdaPi1\n"
          "VariablesToLimit: NumericalCharacteristic\n"
          "TvbConstant: 0.0\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: True");
  const auto muscl =
      TestHelpers::test_creation<grmhd::ValenciaDivClean::Limiters::Minmod>(
          "Type: Muscl\n"
          "VariablesToLimit: NumericalCharacteristic\n"
          "TvbConstant: 0.0\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");

  // Test operators == and !=
  CHECK(lambda_pi1 == lambda_pi1);
  CHECK(lambda_pi1 != lambda_pi1_cons);
  CHECK(lambda_pi1 != lambda_pi1_tvb);
  CHECK(lambda_pi1 != lambda_pi1_noflat);
  CHECK(lambda_pi1 != lambda_pi1_disabled);
  CHECK(lambda_pi1 != muscl);

  // Test that creation from options gives correct object
  const grmhd::ValenciaDivClean::Limiters::Minmod expected_lambda_pi1(
      Limiters::MinmodType::LambdaPi1,
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
          NumericalCharacteristic,
      0., true);
  const grmhd::ValenciaDivClean::Limiters::Minmod expected_lambda_pi1_cons(
      Limiters::MinmodType::LambdaPi1,
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit::Conserved, 0., true);
  const grmhd::ValenciaDivClean::Limiters::Minmod expected_lambda_pi1_tvb(
      Limiters::MinmodType::LambdaPi1,
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
          NumericalCharacteristic,
      1., true);
  const grmhd::ValenciaDivClean::Limiters::Minmod expected_lambda_pi1_noflat(
      Limiters::MinmodType::LambdaPi1,
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
          NumericalCharacteristic,
      0., false);
  const grmhd::ValenciaDivClean::Limiters::Minmod expected_lambda_pi1_disabled(
      Limiters::MinmodType::LambdaPi1,
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
          NumericalCharacteristic,
      0., true, true);
  const grmhd::ValenciaDivClean::Limiters::Minmod expected_muscl(
      Limiters::MinmodType::Muscl,
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
          NumericalCharacteristic,
      0., true);

  CHECK(lambda_pi1 == expected_lambda_pi1);
  CHECK(lambda_pi1_cons == expected_lambda_pi1_cons);
  CHECK(lambda_pi1_tvb == expected_lambda_pi1_tvb);
  CHECK(lambda_pi1_noflat == expected_lambda_pi1_noflat);
  CHECK(lambda_pi1_disabled == expected_lambda_pi1_disabled);
  CHECK(muscl == expected_muscl);
}

void test_neuler_minmod_serialization() noexcept {
  INFO("Testing serialization");
  const grmhd::ValenciaDivClean::Limiters::Minmod minmod(
      Limiters::MinmodType::LambdaPi1,
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
          NumericalCharacteristic,
      1., true);
  test_serialization(minmod);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Limiters.Minmod",
                  "[Limiters][Unit]") {
  test_neuler_minmod_option_parsing();
  test_neuler_minmod_serialization();

  // The package_data function for ValenciaDivCLean::Limiters::Minmod is just a
  // direct call to the generic Minmod package_data. We do not test it.

  // TODO: write this test... hopefully this resembles the NewtonianEuler case,
  // but it will be harder to construct simple inputs in GRMHD so that the
  // characteristic limiting result is predictable.
}
