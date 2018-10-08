// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <memory>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/RegularSphericalWave.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/TestCreation.hpp"

SPECTRE_TEST_CASE("Unit.AnalyticSolutions.WaveEquation.RegularSphericalWave",
                  "[PointwiseFunctions][Unit]") {
  const ScalarWave::Solutions::RegularSphericalWave solution{
      std::make_unique<MathFunctions::Gaussian>(1., 1., 0.)};

  const tnsr::I<DataVector, 3> x{std::array<DataVector, 3>{
      {DataVector({0., 1., 2., 3.}), DataVector({0., 0., 0., 0.}),
       DataVector({0., 0., 0., 0.})}}};
  auto vars = solution.variables(
      x, 1., tmpl::list<ScalarWave::Pi, ScalarWave::Phi<3>, ScalarWave::Psi>{});
  CHECK_ITERABLE_APPROX(
      get<ScalarWave::Psi>(vars).get(),
      DataVector({{1.471517764685769, 0.9816843611112658, 0.1838780156836778,
                   0.006105175451186487}}));
  CHECK_ITERABLE_APPROX(
      get<ScalarWave::Pi>(vars).get(),
      DataVector({{1.471517764685769, -0.07326255555493672, -0.3682496705837024,
                   -0.02442115194544483}}));
  CHECK_ITERABLE_APPROX(
      get<ScalarWave::Phi<3>>(vars).get(0),
      DataVector({{0., -0.9084218055563291, -0.4594482196010212,
                   -0.02645561024157515}}));
  CHECK_ITERABLE_APPROX(get<ScalarWave::Phi<3>>(vars).get(1),
                        DataVector({{0., 0., 0., 0.}}));
  CHECK_ITERABLE_APPROX(get<ScalarWave::Phi<3>>(vars).get(2),
                        DataVector({{0., 0., 0., 0.}}));
  auto dt_vars = solution.variables(
      x, 1.,
      tmpl::list<Tags::dt<ScalarWave::Pi>, Tags::dt<ScalarWave::Phi<3>>,
                 Tags::dt<ScalarWave::Psi>>{});
  CHECK_ITERABLE_APPROX(
      get<Tags::dt<ScalarWave::Psi>>(dt_vars).get(),
      DataVector({{-1.471517764685769, 0.07326255555493672, 0.3682496705837024,
                   0.02442115194544483}}));
  CHECK_ITERABLE_APPROX(
      get<Tags::dt<ScalarWave::Pi>>(dt_vars).get(),
      DataVector({{2.943035529371539, 2.256418944442279, -0.3657814745019688,
                   -0.08547065575381531}}));
  CHECK_ITERABLE_APPROX(get<Tags::dt<ScalarWave::Phi<3>>>(dt_vars).get(0),
                        DataVector({{0., 1.670318500002785, -0.5541022431327671,
                                     -0.09361569118951865}}));
  CHECK_ITERABLE_APPROX(get<Tags::dt<ScalarWave::Phi<3>>>(dt_vars).get(1),
                        DataVector({{0., 0., 0., 0.}}));
  CHECK_ITERABLE_APPROX(get<Tags::dt<ScalarWave::Phi<3>>>(dt_vars).get(2),
                        DataVector({{0., 0., 0., 0.}}));

  const auto created_solution =
      test_creation<ScalarWave::Solutions::RegularSphericalWave>(
          "  Profile:\n"
          "    Gaussian:\n"
          "      Amplitude: 1.\n"
          "      Width: 1.\n"
          "      Center: 0.\n");
  CHECK(
      created_solution.variables(
          x, 1.,
          tmpl::list<ScalarWave::Pi, ScalarWave::Phi<3>, ScalarWave::Psi>{}) ==
      vars);
}
