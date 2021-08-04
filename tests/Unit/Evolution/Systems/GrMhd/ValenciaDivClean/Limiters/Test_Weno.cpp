// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <optional>
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
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/Weno.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoType.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/CharacteristicHelpers.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/Weno.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

void test_neuler_weno_option_parsing() noexcept {
  INFO("Testing option parsing");
  const auto sweno =
      TestHelpers::test_creation<grmhd::ValenciaDivClean::Limiters::Weno>(
          "Type: SimpleWeno\n"
          "VariablesToLimit: NumericalCharacteristic\n"
          "NeighborWeight: 0.001\n"
          "TvbConstant: 0.0\n"
          "KxrcfConstant: None\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto sweno_nw =
      TestHelpers::test_creation<grmhd::ValenciaDivClean::Limiters::Weno>(
          "Type: SimpleWeno\n"
          "VariablesToLimit: NumericalCharacteristic\n"
          "NeighborWeight: 0.02\n"
          "TvbConstant: 0.0\n"
          "KxrcfConstant: None\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto sweno_cons =
      TestHelpers::test_creation<grmhd::ValenciaDivClean::Limiters::Weno>(
          "Type: SimpleWeno\n"
          "VariablesToLimit: Conserved\n"
          "NeighborWeight: 0.001\n"
          "TvbConstant: 0.0\n"
          "KxrcfConstant: None\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto sweno_tvb =
      TestHelpers::test_creation<grmhd::ValenciaDivClean::Limiters::Weno>(
          "Type: SimpleWeno\n"
          "VariablesToLimit: NumericalCharacteristic\n"
          "NeighborWeight: 0.001\n"
          "TvbConstant: 1.0\n"
          "KxrcfConstant: None\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto sweno_noflat =
      TestHelpers::test_creation<grmhd::ValenciaDivClean::Limiters::Weno>(
          "Type: SimpleWeno\n"
          "VariablesToLimit: NumericalCharacteristic\n"
          "NeighborWeight: 0.001\n"
          "TvbConstant: 0.0\n"
          "KxrcfConstant: None\n"
          "ApplyFlattener: False\n"
          "DisableForDebugging: False");
  const auto sweno_disabled =
      TestHelpers::test_creation<grmhd::ValenciaDivClean::Limiters::Weno>(
          "Type: SimpleWeno\n"
          "VariablesToLimit: NumericalCharacteristic\n"
          "NeighborWeight: 0.001\n"
          "TvbConstant: 0.0\n"
          "KxrcfConstant: None\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: True");
  const auto hweno =
      TestHelpers::test_creation<grmhd::ValenciaDivClean::Limiters::Weno>(
          "Type: Hweno\n"
          "VariablesToLimit: NumericalCharacteristic\n"
          "NeighborWeight: 0.001\n"
          "TvbConstant: None\n"
          "KxrcfConstant: 0.0\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");
  const auto hweno_kxrcf =
      TestHelpers::test_creation<grmhd::ValenciaDivClean::Limiters::Weno>(
          "Type: Hweno\n"
          "VariablesToLimit: NumericalCharacteristic\n"
          "NeighborWeight: 0.001\n"
          "TvbConstant: None\n"
          "KxrcfConstant: 1.0\n"
          "ApplyFlattener: True\n"
          "DisableForDebugging: False");

  // Test operators == and !=
  CHECK(sweno == sweno);
  CHECK(sweno != sweno_nw);
  CHECK(sweno != sweno_cons);
  CHECK(sweno != sweno_tvb);
  CHECK(sweno != sweno_noflat);
  CHECK(sweno != sweno_disabled);
  CHECK(sweno != hweno);
  CHECK(hweno != hweno_kxrcf);

  // Test that creation from options gives correct object
  const grmhd::ValenciaDivClean::Limiters::Weno expected_sweno(
      Limiters::WenoType::SimpleWeno,
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
          NumericalCharacteristic,
      0.001, 0.0, {}, true);
  const grmhd::ValenciaDivClean::Limiters::Weno expected_sweno_nw(
      Limiters::WenoType::SimpleWeno,
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
          NumericalCharacteristic,
      0.02, 0.0, {}, true);
  const grmhd::ValenciaDivClean::Limiters::Weno expected_sweno_cons(
      Limiters::WenoType::SimpleWeno,
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit::Conserved, 0.001,
      0.0, {}, true);
  const grmhd::ValenciaDivClean::Limiters::Weno expected_sweno_tvb(
      Limiters::WenoType::SimpleWeno,
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
          NumericalCharacteristic,
      0.001, 1.0, {}, true);
  const grmhd::ValenciaDivClean::Limiters::Weno expected_sweno_noflat(
      Limiters::WenoType::SimpleWeno,
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
          NumericalCharacteristic,
      0.001, 0.0, {}, false);
  const grmhd::ValenciaDivClean::Limiters::Weno expected_sweno_disabled(
      Limiters::WenoType::SimpleWeno,
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
          NumericalCharacteristic,
      0.001, 0.0, {}, true, true);
  const grmhd::ValenciaDivClean::Limiters::Weno expected_hweno(
      Limiters::WenoType::Hweno,
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
          NumericalCharacteristic,
      0.001, {}, 0.0, true);
  const grmhd::ValenciaDivClean::Limiters::Weno expected_hweno_kxrcf(
      Limiters::WenoType::Hweno,
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
          NumericalCharacteristic,
      0.001, {}, 1.0, true);

  CHECK(sweno == expected_sweno);
  CHECK(sweno_nw == expected_sweno_nw);
  CHECK(sweno_cons == expected_sweno_cons);
  CHECK(sweno_tvb == expected_sweno_tvb);
  CHECK(sweno_noflat == expected_sweno_noflat);
  CHECK(sweno_disabled == expected_sweno_disabled);
  CHECK(hweno == expected_hweno);
  CHECK(hweno_kxrcf == expected_hweno_kxrcf);
}

void test_neuler_weno_serialization() noexcept {
  INFO("Testing serialization");
  const grmhd::ValenciaDivClean::Limiters::Weno weno(
      Limiters::WenoType::SimpleWeno,
      grmhd::ValenciaDivClean::Limiters::VariablesToLimit::
          NumericalCharacteristic,
      0.001, 1., {}, true);
  test_serialization(weno);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Limiters.Weno",
                  "[Limiters][Unit]") {
  test_neuler_weno_option_parsing();
  test_neuler_weno_serialization();

  // The package_data function for ValenciaDivClean::Limiters::Weno is just a
  // direct call to the generic Weno package_data. We do not test it.

  // TODO: write this test... hopefully this resembles the NewtonianEuler case,
  // but it will be harder to construct simple inputs in GRMHD so that the
  // characteristic limiting result is predictable.
}
