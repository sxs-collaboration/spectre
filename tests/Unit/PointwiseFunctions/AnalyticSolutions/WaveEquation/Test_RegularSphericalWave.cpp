// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <memory>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"  // IWYU pragma: keep
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/RegularSphericalWave.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   tmpl::list<MathFunctions::Gaussian<1, Frame::Inertial>>>,
        tmpl::pair<evolution::initial_data::InitialData,
                   tmpl::list<ScalarWave::Solutions::RegularSphericalWave>>>;
  };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.AnalyticSolutions.WaveEquation.RegularSphericalWave",
                  "[PointwiseFunctions][Unit]") {
  register_factory_classes_with_charm<Metavariables>();
  const ScalarWave::Solutions::RegularSphericalWave solution{
      std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(1., 1.,
                                                                    0.)};
  CHECK_FALSE(solution != solution);
  test_copy_semantics(solution);
  auto solution_for_move = solution;
  test_move_semantics(std::move(solution_for_move), solution);

  const tnsr::I<DataVector, 3> x{std::array<DataVector, 3>{
      {DataVector({0., 1., 2., 3.}), DataVector({0., 0., 0., 0.}),
       DataVector({0., 0., 0., 0.})}}};
  auto vars =
      solution.variables(x, 1.,
                         tmpl::list<ScalarWave::Tags::Psi, ScalarWave::Tags::Pi,
                                    ScalarWave::Tags::Phi<3>>{});
  CHECK_ITERABLE_APPROX(
      get<ScalarWave::Tags::Psi>(vars).get(),
      DataVector({{1.471517764685769, 0.9816843611112658, 0.1838780156836778,
                   0.006105175451186487}}));
  CHECK_ITERABLE_APPROX(
      get<ScalarWave::Tags::Pi>(vars).get(),
      DataVector({{1.471517764685769, -0.07326255555493672, -0.3682496705837024,
                   -0.02442115194544483}}));
  CHECK_ITERABLE_APPROX(
      get<ScalarWave::Tags::Phi<3>>(vars).get(0),
      DataVector({{0., -0.9084218055563291, -0.4594482196010212,
                   -0.02645561024157515}}));
  CHECK_ITERABLE_APPROX(get<ScalarWave::Tags::Phi<3>>(vars).get(1),
                        DataVector({{0., 0., 0., 0.}}));
  CHECK_ITERABLE_APPROX(get<ScalarWave::Tags::Phi<3>>(vars).get(2),
                        DataVector({{0., 0., 0., 0.}}));
  auto dt_vars =
      solution.variables(x, 1.,
                         tmpl::list<Tags::dt<ScalarWave::Tags::Psi>,
                                    Tags::dt<ScalarWave::Tags::Pi>,
                                    Tags::dt<ScalarWave::Tags::Phi<3>>>{});
  CHECK_ITERABLE_APPROX(
      get<Tags::dt<ScalarWave::Tags::Psi>>(dt_vars).get(),
      DataVector({{-1.471517764685769, 0.07326255555493672, 0.3682496705837024,
                   0.02442115194544483}}));
  CHECK_ITERABLE_APPROX(
      get<Tags::dt<ScalarWave::Tags::Pi>>(dt_vars).get(),
      DataVector({{2.943035529371539, 2.256418944442279, -0.3657814745019688,
                   -0.08547065575381531}}));
  CHECK_ITERABLE_APPROX(get<Tags::dt<ScalarWave::Tags::Phi<3>>>(dt_vars).get(0),
                        DataVector({{0., 1.670318500002785, -0.5541022431327671,
                                     -0.09361569118951865}}));
  CHECK_ITERABLE_APPROX(get<Tags::dt<ScalarWave::Tags::Phi<3>>>(dt_vars).get(1),
                        DataVector({{0., 0., 0., 0.}}));
  CHECK_ITERABLE_APPROX(get<Tags::dt<ScalarWave::Tags::Phi<3>>>(dt_vars).get(2),
                        DataVector({{0., 0., 0., 0.}}));

  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag<
          evolution::initial_data::OptionTags::InitialData, Metavariables>(
          "RegularSphericalWave:\n"
          "  Profile:\n"
          "    Gaussian:\n"
          "      Amplitude: 1.\n"
          "      Width: 1.\n"
          "      Center: 0.\n")
          ->get_clone();
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& created_solution =
      dynamic_cast<const ScalarWave::Solutions::RegularSphericalWave&>(
          *deserialized_option_solution);

  CHECK(created_solution.variables(
            x, 1.,
            tmpl::list<ScalarWave::Tags::Psi, ScalarWave::Tags::Pi,
                       ScalarWave::Tags::Phi<3>>{}) == vars);
}
