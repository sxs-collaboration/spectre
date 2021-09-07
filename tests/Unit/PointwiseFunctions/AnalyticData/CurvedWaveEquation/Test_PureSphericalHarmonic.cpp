// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/AnalyticData/CurvedWaveEquation/PureSphericalHarmonic.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
void test_create_from_options() {
  INFO("Testing option creation");
  const auto option_parsed = TestHelpers::test_creation<
      CurvedScalarWave::AnalyticData::PureSphericalHarmonic>(
      "Radius: 0.8 \n"
      "Width: 2. \n"
      "Mode: [10, 8]");
  const CurvedScalarWave::AnalyticData::PureSphericalHarmonic constructed{
      0.8, 2., {10, 8}};
  CHECK(option_parsed == constructed);
  CHECK_FALSE(option_parsed != constructed);

  INFO("Testing serialization");
  test_serialization(constructed);
}

// `check_with_random_values` does not allow us to forward the l and m arguments
// of type size_t / int to python, so we generate the random numbers and call
// the python function manually here
void test_variables() {
  INFO("Testing random values against python function");
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> x_dist{1., 10.};
  std::uniform_real_distribution<double> radius_dist{1., 10.};
  std::uniform_real_distribution<double> width_dist{2., 20.};
  // size of the DataVector we are testing with
  const size_t dv_size = 10;
  for (size_t l = 0; l < 8; l++) {
    for (int m = -l; m <= static_cast<int>(l); ++m) {
      const auto x = make_with_random_values<tnsr::I<DataVector, 3>>(
          make_not_null(&gen), make_not_null(&x_dist), DataVector(dv_size));
      const double radius = radius_dist(gen);
      const double width = width_dist(gen);
      const auto py_res = pypp::call<Scalar<DataVector>>(
          "PureSphericalHarmonic", "pi", x, radius, width, l, m);

      const CurvedScalarWave::AnalyticData::PureSphericalHarmonic sh{
          radius, width, {l, m}};
      const auto sh_res = sh.variables(x, 0., {});

      CHECK_ITERABLE_APPROX(get<CurvedScalarWave::Pi>(sh_res), py_res);
      CHECK(get<CurvedScalarWave::Phi<3>>(sh_res) ==
            make_with_value<tnsr::i<DataVector, 3>>(dv_size, 0.));
      CHECK(get<CurvedScalarWave::Psi>(sh_res) ==
            make_with_value<Scalar<DataVector>>(dv_size, 0.));
    }
  }
}

void test_errors() {
  INFO("Testing configuration errors");
  CHECK_THROWS_WITH(CurvedScalarWave::AnalyticData::PureSphericalHarmonic(
                        -0.1, 1., {1, 1}, Options::Context{false, {}, 1, 1}),
                    Catch::Matchers::Contains(
                        "The radius must be greater than 0 but is -0.1"));

  CHECK_THROWS_WITH(CurvedScalarWave::AnalyticData::PureSphericalHarmonic(
                        1., -0.1, {1, 1}, Options::Context{false, {}, 1, 1}),
                    Catch::Matchers::Contains(
                        "The width must be greater than 0 but is -0.1"));

  CHECK_THROWS_WITH(
      CurvedScalarWave::AnalyticData::PureSphericalHarmonic(
          1., 1., {7, 8}, Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "The absolute value of the m_mode must be less than or equal to the "
          "l-mode but the m-mode is 8 and the l-mode is 7"));
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.CurvedWaveEquation.SphericalHarmonic",
    "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/CurvedWaveEquation"};
  test_create_from_options();
  test_variables();
  test_errors();
}
